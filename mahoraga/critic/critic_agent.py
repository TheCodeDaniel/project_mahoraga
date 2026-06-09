import re
from urllib.parse import quote_plus

_NEGATION_RE = re.compile(
    r"\b(not|no\b|never|doesn't|does not|don't|cannot|can't|isn't|is not|won't|wouldn't)\b",
    re.IGNORECASE,
)

import requests
from bs4 import BeautifulSoup


_DDG_API_URL = "https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
_DDG_HTML_URL = "https://html.duckduckgo.com/html/?q={query}"
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def _claim_words(claim: str) -> set[str]:
    return {w.lower() for w in re.findall(r"\w+", claim)}


def _positive_claim_words(claim: str) -> set[str]:
    """Claim words with negation particles stripped, for positive-form matching."""
    return _claim_words(_NEGATION_RE.sub(" ", claim))


def _relevance(claim_words: set[str], text: str) -> float:
    if not text or not claim_words:
        return 0.0
    text_words = {w.lower() for w in re.findall(r"\w+", text)}
    return len(claim_words & text_words) / len(claim_words)


def _is_challenge_page(resp: requests.Response) -> bool:
    """Return True if DuckDuckGo returned a bot-challenge page instead of results."""
    return resp.status_code == 202 or "anomaly.js" in resp.text


def _ddg_api_sources(claim: str) -> list[dict]:
    """Step 1: DuckDuckGo Instant Answer API."""
    sources: list[dict] = []
    try:
        url = _DDG_API_URL.format(query=quote_plus(claim))
        resp = requests.get(url, timeout=8)
        if _is_challenge_page(resp):
            return sources
        data = resp.json()

        abstract = data.get("AbstractText", "").strip()
        abstract_url = data.get("AbstractURL", "").strip()
        if abstract:
            sources.append({"url": abstract_url, "snippet": abstract})

        for topic in data.get("RelatedTopics", [])[:4]:
            text = topic.get("Text", "").strip()
            first_url = topic.get("FirstURL", "").strip()
            if text:
                sources.append({"url": first_url, "snippet": text})
    except Exception:
        pass
    return sources


def _ddg_html_sources(claim: str) -> list[dict]:
    """Step 2: DuckDuckGo HTML search, with ddgs fallback on bot-challenge."""
    sources: list[dict] = []
    try:
        url = _DDG_HTML_URL.format(query=quote_plus(claim))
        resp = requests.get(url, headers=_BROWSER_HEADERS, timeout=8)

        if not _is_challenge_page(resp):
            soup = BeautifulSoup(resp.text, "html.parser")
            for result in soup.select(".result")[:5]:
                snippet_tag = result.select_one(".result__snippet")
                url_tag = result.select_one(".result__url")
                snippet = snippet_tag.get_text(" ", strip=True) if snippet_tag else ""
                href = url_tag.get_text(strip=True) if url_tag else ""
                if snippet:
                    sources.append({"url": href, "snippet": snippet})
            if sources:
                return sources

        # Fallback: ddgs library handles DDG session tokens transparently
        from ddgs import DDGS  # lazy import — only needed when DDG blocks direct requests

        for r in DDGS().text(claim, max_results=5):
            snippet = r.get("body", "").strip()
            href = r.get("href", "").strip()
            if snippet:
                sources.append({"url": href, "snippet": snippet})
    except Exception:
        pass
    return sources


class CriticAgent:
    def evaluate(self, claim: str) -> dict:
        words = _claim_words(claim)

        # Steps 1 & 2: gather sources
        sources = _ddg_api_sources(claim) + _ddg_html_sources(claim)

        # Step 3: score each source by word-overlap relevance
        scored = [
            {**src, "_score": _relevance(words, src["snippet"])}
            for src in sources
        ]
        scored.sort(key=lambda x: x["_score"], reverse=True)

        # Step 4: confidence = average of top-3 scores
        top3 = scored[:3]
        confidence = round(sum(s["_score"] for s in top3) / len(top3), 2) if top3 else 0.0

        # Step 5: verdict
        if confidence >= 0.6:
            verdict: bool | None = True
        elif confidence < 0.3:
            verdict = False
        else:
            verdict = None

        # Step 6: correction
        correction: str | None = None
        if verdict is False and scored:
            correction = scored[0]["snippet"]

        # Negation override: if the claim negates something and sources
        # strongly confirm the positive form, the claim is false.
        if _NEGATION_RE.search(claim) and sources:
            pos_words = _positive_claim_words(claim)
            pos_scored = sorted(sources, key=lambda s: _relevance(pos_words, s["snippet"]), reverse=True)
            pos_top3 = pos_scored[:3]
            pos_conf = sum(_relevance(pos_words, s["snippet"]) for s in pos_top3) / len(pos_top3)
            if pos_conf >= 0.5:
                verdict = False
                confidence = round(pos_conf, 2)
                correction = pos_scored[0]["snippet"]

        clean_sources = [{"url": s["url"], "snippet": s["snippet"]} for s in scored]

        return {
            "verdict": verdict,
            "confidence": confidence,
            "sources": clean_sources,
            "correction": correction,
            "claim": claim,
        }


if __name__ == "__main__":
    import json

    agent = CriticAgent()
    claims = [
        "The flutter_gemma package requires iOS 14 or higher",
        "Python was created by Guido van Rossum in 1991",
    ]
    for c in claims:
        print(f"\nClaim: {c}")
        result = agent.evaluate(c)
        print(json.dumps(result, indent=2))
