"""Concurrent HTML scraper using asyncio.to_thread — no new dependencies."""
import asyncio
import logging
import re

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

MAX_CHARS = 10_000
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
}


def _fetch_sync(url: str) -> dict | None:
    try:
        r = requests.get(url, headers=_HEADERS, timeout=15, allow_redirects=True)
        r.raise_for_status()
        if "text/html" not in r.headers.get("Content-Type", ""):
            return None
        soup = BeautifulSoup(r.content, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
            tag.decompose()
        title_el = soup.find("title")
        title = title_el.get_text(strip=True) if title_el else url
        text = re.sub(r"\n{3,}", "\n\n", soup.get_text(separator="\n", strip=True))
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS] + "\n\n[... truncated ...]"
        return {"title": title, "text": text} if text.strip() else None
    except Exception as e:
        logger.debug("Fetch failed %s: %s", url, e)
        return None


async def fetch_all(urls: list[str], max_concurrent: int = 5) -> list[dict]:
    """Fetch all URLs concurrently, respecting max_concurrent limit."""
    sem = asyncio.Semaphore(max_concurrent)

    async def _one(url: str) -> dict:
        async with sem:
            content = await asyncio.to_thread(_fetch_sync, url)
        return {"url": url, "content": content}

    return await asyncio.gather(*(_one(u) for u in urls))
