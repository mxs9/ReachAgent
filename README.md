# ReachAgent

An agentic pipeline that researches a target person on the web and writes a personalized cold outreach email in your voice, grounded in real signals rather than templates.

Upload your resume, point it at someone, and get a draft that actually references their work.

Built on [PocketFlow](https://github.com/The-Pocket/PocketFlow) by Zachary Huang.

---

## How it works

```
Search → Fetch (parallel) → Analyze → Draft → Eval
                                         ↑       |
                                         └─retry─┘
```

| Node | What it does |
|------|-------------|
| **SearchNode** | Queries DuckDuckGo for public info about the target. No search API key required. |
| **FetchNode** | Fetches all result pages concurrently using `asyncio.to_thread` + a semaphore. ~5x faster than sequential. |
| **AnalyzeNode** | Runs each page through the LLM in parallel. Uses structured tool calls to extract personalization signals reliably — no YAML parsing, no regex. |
| **DraftNode** | Writes a cold email grounded in the extracted signals and your resume. Picks the 2-3 most relevant credentials for this specific person. |
| **EvalNode** | Scores the draft on specificity, authenticity, and conciseness (0-10 each). If the total falls below 21/30, it passes a one-sentence critique back to DraftNode and retries. Max 2 retries. |

---

## Setup

Requires Python 3.11+

```bash
git clone https://github.com/your-username/ReachAgent
cd ReachAgent
pip install -r requirements.txt
```

Create a `.env` file:

```env
# Pick one. DeepSeek is recommended: fast and inexpensive for this task.
DEEPSEEK_API_KEY=sk-...

# OpenAI also works
# OPENAI_API_KEY=sk-...

# As does Anthropic
# ANTHROPIC_API_KEY=sk-ant-...

# Optional: force a specific provider when multiple keys are set
# LLM_PROVIDER=deepseek
```

Provider auto-detection order: DeepSeek > OpenAI > Anthropic.

---

## Usage

Open `flow.py` and edit the `shared` input in `main()`:

```python
shared = {
    "input": {
        # Target person
        "first_name": "Jane",
        "last_name":  "Smith",
        "keywords":   "Anthropic research engineering",

        # Your resume as plain text (recommended)
        # The LLM picks the 2-3 most relevant parts for this specific outreach
        "resume": open("my_resume.txt").read(),

        # Or a short bio if you prefer
        # "sender_bio": "MS in CS from Stanford, 2 years building ML infra at ...",

        # What you are looking for
        "target_role": "a research engineering role at Anthropic",

        # What signals to look for about the target
        "personalization_factors": [
            {
                "name":        "recent_work",
                "description": "Recent projects, papers, or conference talks",
                "action":      "Reference it specifically: 'I read your post on [X]...'"
            },
            {
                "name":        "shared_interest",
                "description": "Overlapping technical interests",
                "action":      "Name the concrete overlap with your own background"
            },
            {
                "name":        "alumni_connection",
                "description": "Shared university, lab, or advisor",
                "action":      "Mention it naturally, not as the main hook"
            },
        ],
    }
}
```

Then run:

```bash
python flow.py
```

The email is printed to stdout along with the eval score and which signals were used.

---

## Design decisions

**Structured outputs over text parsing.**
Asking the LLM to return YAML and then parsing it breaks whenever the format slips. Tool calls enforce the schema at the API level — the output is always a valid dict.

**Concurrent fetching with no new dependencies.**
`asyncio.to_thread` wraps the existing `requests`-based scraper and runs it in a thread pool. A semaphore limits concurrency. This cuts wall-clock fetch time from ~20s to ~3s for 10 URLs without adding `aiohttp`.

**Resume-aware drafting.**
Instead of a hardcoded bio, the full resume is passed to the LLM, which selects the most relevant 2-3 credentials for each specific target. The same resume produces different openings depending on who you are writing to.

**Self-correcting eval loop.**
A single pass produces inconsistent quality. `EvalNode` scores each draft and returns a one-sentence critique when the score is too low, which is injected into the next draft prompt. The loop caps at 2 retries to keep latency predictable.

**Multi-provider support.**
`utils/llm.py` exposes two functions (`call_llm` and `call_llm_structured`) that route to DeepSeek, OpenAI, or Anthropic. Switching providers is a one-line change in `.env`.

---

## Project structure

```
ReachAgent/
├── flow.py           # pipeline: 5 nodes, ~220 lines
├── pocketflow/       # async workflow engine by Zachary Huang (~100 lines)
└── utils/
    ├── llm.py        # multi-provider client: DeepSeek / OpenAI / Anthropic
    ├── search.py     # DuckDuckGo search, no API key required
    └── scraper.py    # concurrent HTML fetcher using asyncio.to_thread
```

---

## Requirements

```
openai>=1.0.0
anthropic>=0.40.0
requests>=2.31.0
beautifulsoup4>=4.12.0
ddgs>=6.0
python-dotenv>=1.0.0
pyyaml>=6.0
```

---

## Acknowledgements

[PocketFlow](https://github.com/The-Pocket/PocketFlow) was created by [Zachary Huang](https://github.com/The-Pocket). Its minimal design — prep, exec, post, and nothing else — makes it straightforward to reason about what each node does and to test nodes in isolation. ReachAgent would not be as clean without it.
