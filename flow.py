"""
Cold Outreach Email Generator
==============================
Researches a target person on the web and generates a personalized cold email.

Pipeline
--------
SearchNode → FetchNode → AnalyzeNode → DraftNode → EvalNode
                                            ↑            |
                                            └── retry ───┘ (max 2x)

Key design choices
------------------
- Structured LLM outputs (tool calls) instead of fragile YAML parsing
- Parallel web fetching with asyncio.to_thread + semaphore
- Self-correcting loop: EvalNode scores the draft and retries if below threshold
- Multi-provider: DeepSeek / OpenAI / Anthropic, auto-detected from .env
"""
import asyncio
import logging
import sys

from pocketflow import AsyncFlow, AsyncNode, AsyncParallelBatchNode

from utils.llm import call_llm_structured
from utils.scraper import fetch_all
from utils.search import search_web

# ── Logging ───────────────────────────────────────────────────────────────────

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import warnings
warnings.filterwarnings("ignore", message="Flow ends.*pass.*")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("cold_reach")

# ── LLM output schemas ────────────────────────────────────────────────────────

_ANALYZE_SCHEMA = {
    "type": "object",
    "properties": {
        "factors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name":       {"type": "string"},
                    "actionable": {"type": "boolean"},
                    "details":    {"type": "string"},
                },
                "required": ["name", "actionable", "details"],
            },
        }
    },
    "required": ["factors"],
}

_DRAFT_SCHEMA = {
    "type": "object",
    "properties": {
        "subject": {
            "type": "string",
            "description": "Subject line: specific to this person or their team, under 10 words",
        },
        "opening": {
            "type": "string",
            "description": (
                "1-2 sentences, under 35 words. Reference something real about their work or company. "
                "Do not start with 'I'. No em dashes."
            ),
        },
        "body": {
            "type": "string",
            "description": (
                "2-3 short paragraphs, 120-180 words total. "
                "Briefly introduce who you are and your most relevant credential. "
                "Connect your background to what they are working on. "
                "Sound like a real person, not a cover letter. No em dashes."
            ),
        },
        "closing": {
            "type": "string",
            "description": (
                "1-2 sentences. Ask for a 20-minute chat or to be considered for a specific role. "
                "Be direct. No 'let me know if you have time', no 'at your earliest convenience'."
            ),
        },
    },
    "required": ["subject", "opening", "body", "closing"],
}

_EVAL_SCHEMA = {
    "type": "object",
    "properties": {
        "specificity": {
            "type": "integer",
            "description": "0-10: does it reference concrete details about this person, or could it be sent to anyone?",
        },
        "authenticity": {
            "type": "integer",
            "description": "0-10: genuine and human vs. templated and salesy",
        },
        "conciseness": {
            "type": "integer",
            "description": "0-10: right length, no filler phrases",
        },
        "feedback": {
            "type": "string",
            "description": "One sentence on the most important thing to improve.",
        },
    },
    "required": ["specificity", "authenticity", "conciseness", "feedback"],
}

PASS_SCORE   = 21   # out of 30; below this triggers a retry
MAX_RETRIES  = 2    # max draft regeneration attempts


# ── Nodes ─────────────────────────────────────────────────────────────────────

class SearchNode(AsyncNode):
    """Search the web for publicly available info about the target person."""

    async def prep_async(self, shared: dict) -> str:
        inp = shared["input"]
        return f"{inp['first_name']} {inp['last_name']} {inp.get('keywords', '')}"

    async def exec_async(self, query: str) -> list[dict]:
        logger.info("Searching: %r", query)
        return await asyncio.to_thread(search_web, query)

    async def post_async(self, shared: dict, prep_res, exec_res: list[dict]) -> str:
        shared["search_results"] = exec_res
        logger.info("%d search results", len(exec_res))
        return "default"


class FetchNode(AsyncNode):
    """Fetch all search result pages concurrently."""

    async def prep_async(self, shared: dict) -> list[str]:
        return [r["link"] for r in shared["search_results"] if "link" in r]

    async def exec_async(self, urls: list[str]) -> list[dict]:
        logger.info("Fetching %d URLs in parallel", len(urls))
        return await fetch_all(urls)

    async def post_async(self, shared: dict, prep_res, exec_res: list[dict]) -> str:
        valid = [r for r in exec_res if r["content"]]
        shared["web_contents"] = valid
        logger.info("%d/%d pages fetched successfully", len(valid), len(exec_res))
        return "default"


class AnalyzeNode(AsyncParallelBatchNode):
    """
    Extract personalization signals from each page in parallel.
    Uses structured tool calls — no YAML parsing, no fragile text extraction.
    """

    async def prep_async(self, shared: dict) -> list[dict]:
        self._inp     = shared["input"]
        self._factors = shared["input"]["personalization_factors"]
        return shared["web_contents"]

    async def exec_async(self, page: dict) -> dict:
        factor_list = "\n".join(
            f"  - {f['name']}: {f['description']}" for f in self._factors
        )
        prompt = (
            f"Analyze this webpage about {self._inp['first_name']} {self._inp['last_name']}.\n\n"
            f"URL: {page['url']}\n"
            f"Title: {page['content']['title']}\n\n"
            f"Text:\n{page['content']['text']}\n\n"
            f"Extract ONLY these personalization factors if clearly evidenced in the text:\n{factor_list}\n\n"
            "Set actionable=true only when you see direct evidence. Do not infer or hallucinate."
        )
        try:
            return await asyncio.to_thread(
                call_llm_structured, prompt, _ANALYZE_SCHEMA, "extract_personalization"
            )
        except Exception as e:
            logger.debug("Analysis failed for %s: %s", page["url"], e)
            return {"factors": []}

    async def exec_fallback_async(self, page: dict, exc: Exception) -> dict:
        logger.debug("Analysis fallback for %s: %s", page.get("url", "?"), exc)
        return {"factors": []}

    async def post_async(self, shared: dict, prep_res, analyses: list[dict]) -> str:
        # Merge factors across pages; keep first actionable evidence found per factor
        merged: dict[str, dict] = {}
        for analysis in analyses:
            for f in analysis.get("factors", []):
                name = f.get("name", "")
                if f.get("actionable") and f.get("details") and name not in merged:
                    action = next(
                        (x["action"] for x in self._factors if x["name"] == name), ""
                    )
                    merged[name] = {"details": f["details"], "action": action}

        shared["personalization"] = merged
        logger.info("%d actionable personalization signals found", len(merged))
        return "default"


class DraftNode(AsyncNode):
    """Generate a personalized cold email using structured LLM output."""

    async def prep_async(self, shared: dict) -> tuple:
        inp        = shared["input"]
        signals    = shared.get("personalization", {})
        feedback   = shared.get("eval_feedback", "")

        signal_str = "\n".join(
            f"  - {name}: {v['details']} | suggested angle: {v['action']}"
            for name, v in signals.items()
        ) or "  (no specific signals found; write a thoughtful opener based on their public work)"

        return inp, signal_str, feedback

    async def exec_async(self, prep_data: tuple) -> dict:
        inp, signal_str, feedback = prep_data
        person = f"{inp['first_name']} {inp['last_name']}"
        role   = inp.get("target_role", "a relevant role at their company")

        # Accept either a full resume (preferred) or a short bio fallback
        resume = inp.get("resume", "").strip()
        bio    = inp.get("sender_bio", "").strip()
        if resume:
            sender_context = (
                f"Here is the sender's resume. Read it carefully and pick the 2-3 most relevant "
                f"experiences or credentials for this specific outreach. Do not list everything.\n\n"
                f"{resume}"
            )
        elif bio:
            sender_context = f"About the sender: {bio}"
        else:
            sender_context = "About the sender: (not provided — write a placeholder)"

        retry_note = (
            f"\n\nThe previous draft was sent back. Fix this specifically: {feedback}"
            if feedback else ""
        )
        prompt = (
            f"Write a cold outreach email from a job seeker to {person}.\n\n"
            f"{sender_context}\n\n"
            f"What the sender is looking for: {role}\n\n"
            f"Things found about {person} that can be referenced:\n{signal_str}\n\n"
            "Guidelines:\n"
            "1. Opening: reference one specific thing you actually read or noticed about their work. "
            "Not a sweeping compliment like 'your work sits exactly at the intersection of X and Y'. "
            "Instead, name the specific piece, talk, or project and say what specifically caught your attention. "
            "Do not start with 'I'. Never say 'I came across your profile'.\n"
            "2. Body: introduce the sender in 1-2 sentences using their actual background, "
            "then connect a specific skill or project to what this person works on. "
            "When stating career intent, frame it as exploring or transitioning, not asking for a job. "
            "Use 'I am exploring roles focused on [specific area]' rather than 'I am looking for a role at your company'. "
            "Name something concrete the sender could contribute: tooling, audits, evaluation frameworks, "
            "model governance infrastructure, etc. Under 180 words total.\n"
            "3. Write in the sender's authentic voice. Match the level of formality to their background.\n"
            "4. Closing: frame it as a genuine conversation request, not a job ask. "
            "Something like: 'Would you be open to a 20-minute conversation? "
            "I would be glad to hear your perspective on [specific topic] and where someone with my background might contribute.' "
            "Warm and direct, not transactional.\n"
            "5. Do not use em dashes anywhere in the email.\n"
            "6. Banned: 'passionate', 'excited to', 'thrilled', 'I am writing to', "
            "'leverage', 'synergy', 'touch base', 'circle back', 'delve', "
            "'at your earliest convenience', 'I would love to', 'it would be an honor', "
            "'the exact space where I want to contribute', 'exactly where X meets Y'."
            f"{retry_note}"
        )
        return await asyncio.to_thread(
            call_llm_structured, prompt, _DRAFT_SCHEMA, "draft_email"
        )

    async def post_async(self, shared: dict, prep_res, exec_res: dict) -> str:
        shared["draft"] = exec_res
        return "default"


class EvalNode(AsyncNode):
    """
    Score the draft on three dimensions (0-10 each).
    Returns 'retry' if total < PASS_SCORE and retries remain, else 'pass'.
    """

    async def prep_async(self, shared: dict) -> tuple:
        return shared["draft"], shared["input"]

    async def exec_async(self, prep_data: tuple) -> dict:
        draft, inp = prep_data
        person = f"{inp['first_name']} {inp['last_name']}"
        email_text = (
            f"Subject: {draft.get('subject', '')}\n\n"
            f"{draft.get('opening', '')}\n\n"
            f"{draft.get('body', '')}\n\n"
            f"{draft.get('closing', '')}"
        )
        prompt = (
            f"Evaluate this job-seeker cold outreach email to {person}.\n\n"
            f"{email_text}\n\n"
            "Score each dimension 0-10:\n"
            "- specificity: does it reference real, concrete details about this person's work? "
            "Deduct points for sweeping openers like 'your work sits exactly at the intersection of X and Y' "
            "or vague praise that doesn't show the sender actually read something specific. "
            "An email that could be sent to anyone scores 3 or below.\n"
            "- authenticity: does it sound like a real person wrote it? "
            "Deduct for em dashes, overly formal phrasing, or banned phrases like "
            "'passionate', 'leverage', 'I would love to', 'it would be an honor'. "
            "Also deduct if the closing sounds like a job request rather than a genuine conversation ask. "
            "Also deduct if the career intent is stated too directly ('I am looking for a role at your company').\n"
            "- conciseness: under 200 words with no filler? "
            "Deduct for sentences that add no information.\n\n"
            "Be strict. Most AI-generated emails score 4-6. A good human email scores 7-9."
        )
        return await asyncio.to_thread(
            call_llm_structured, prompt, _EVAL_SCHEMA, "evaluate_email"
        )

    async def post_async(self, shared: dict, prep_res, exec_res: dict) -> str:
        s = exec_res.get("specificity",  0)
        a = exec_res.get("authenticity", 0)
        c = exec_res.get("conciseness",  0)
        total   = s + a + c
        retries = shared.get("draft_retries", 0)

        logger.info(
            "Eval — specificity=%d  authenticity=%d  conciseness=%d  total=%d/30",
            s, a, c, total,
        )

        if total >= PASS_SCORE or retries >= MAX_RETRIES:
            shared["eval"] = exec_res
            if total < PASS_SCORE:
                logger.info("Max retries reached — using best available draft")
            return "pass"

        logger.info("Draft rejected (score %d < %d) — retrying. Feedback: %s",
                    total, PASS_SCORE, exec_res.get("feedback", ""))
        shared["eval_feedback"]  = exec_res.get("feedback", "")
        shared["draft_retries"]  = retries + 1
        return "retry"


# ── Flow ──────────────────────────────────────────────────────────────────────

def build_flow() -> AsyncFlow:
    search  = SearchNode()
    fetch   = FetchNode()
    analyze = AnalyzeNode()
    draft   = DraftNode()
    eval_   = EvalNode()

    search >> fetch >> analyze >> draft >> eval_
    (eval_ - "retry") >> draft  # self-correcting loop; "pass" ends the flow

    return AsyncFlow(start=search)


# ── Output ────────────────────────────────────────────────────────────────────

def _print_results(shared: dict) -> None:
    draft  = shared.get("draft", {})
    eva    = shared.get("eval", {})
    inp    = shared["input"]
    person = f"{inp['first_name']} {inp['last_name']}"

    print(f"\n{'═' * 60}")
    print(f"  Cold Outreach — {person}")
    print(f"{'═' * 60}")

    if eva:
        total = (eva.get("specificity", 0) + eva.get("authenticity", 0)
                 + eva.get("conciseness", 0))
        print(f"\n  Quality  {total}/30  "
              f"(specificity={eva.get('specificity',0)}  "
              f"authenticity={eva.get('authenticity',0)}  "
              f"conciseness={eva.get('conciseness',0)})")

    print(f"\n  Subject: {draft.get('subject', '')}\n")
    print(draft.get("opening", ""))
    print()
    print(draft.get("body", ""))
    print()
    print(draft.get("closing", ""))

    signals = shared.get("personalization", {})
    if signals:
        print(f"\n{'─' * 60}")
        print(f"  Signals used ({len(signals)}):")
        for name, v in signals.items():
            print(f"    • {name}: {v['details'][:90]}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    # ── Configure your outreach here ─────────────────────────────────────────
    #
    # Option A: paste your resume as plain text (recommended)
    #   shared["input"]["resume"] = open("my_resume.txt").read()
    #
    # Option B: write a short bio instead
    #   shared["input"]["sender_bio"] = "PhD in CS from MIT, 3 years at Google..."
    #
    shared = {
        "input": {
            "first_name": "Jane",
            "last_name":  "Smith",
            "keywords":   "Anthropic research engineering",

            # Paste your resume text here, or load from a file (see above)
            "resume": """
[Paste your resume here as plain text]

Name: Your Name
Education: ...
Experience: ...
Skills: ...
""",
            # What you are looking for at their company
            "target_role": "a research engineering or applied AI role",

            # What signals to look for about the target person
            "personalization_factors": [
                {
                    "name":        "recent_work",
                    "description": "Recent projects, papers, blog posts, or conference talks",
                    "action":      "Reference the specific work: 'I read your post on [X]...'"
                },
                {
                    "name":        "team_focus",
                    "description": "The team or problem area they focus on",
                    "action":      "Connect your background to their problem area directly"
                },
                {
                    "name":        "shared_interest",
                    "description": "Overlapping technical interests in AI, ML, or the domain they work in",
                    "action":      "Name the concrete overlap: 'Your work on [X] is close to what I built...'"
                },
                {
                    "name":        "alumni_connection",
                    "description": "Shared university, lab, program, or advisor",
                    "action":      "Mention it naturally, not as the main hook"
                },
            ],
        }
    }

    flow = build_flow()
    asyncio.run(flow.run_async(shared))
    _print_results(shared)


if __name__ == "__main__":
    main()
