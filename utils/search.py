"""DuckDuckGo web search — no API key required."""
import logging

logger = logging.getLogger(__name__)


def search_web(query: str, max_results: int = 10) -> list[dict]:
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS

    results = []
    with DDGS() as ddgs:
        for item in ddgs.text(query, max_results=max_results):
            href = item.get("href") or item.get("url")
            if href:
                results.append({
                    "link":    href,
                    "title":   item.get("title", ""),
                    "snippet": item.get("body", ""),
                })

    logger.debug("%d results for %r", len(results), query)
    return results
