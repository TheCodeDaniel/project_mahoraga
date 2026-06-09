from __future__ import annotations

from urllib.parse import quote_plus

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

# Lazily loaded on first use — avoids slow import at startup
_nli_pipeline = None


def _get_nli():
    global _nli_pipeline
    if _nli_pipeline is None:
        from transformers import pipeline
        _nli_pipeline = pipeline(
            "zero-shot-classification",
            model="cross-encoder/nli-deberta-v3-small",
        )
    return _nli_pipeline


def _is_challenge_page(resp: requests.Response) -> bool:
    return resp.status_code == 202 or "anomaly.js" in resp.text


def _ddg_api_sources(claim: str) -> list[dict]:
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

        from ddgs import DDGS
        for r in DDGS().text(claim, max_results=5):
            snippet = r.get("body", "").strip()
            href = r.get("href", "").strip()
            if snippet:
                sources.append({"url": href, "snippet": snippet})
    except Exception:
        pass
    return sources


def _nli_score(claim: str, snippet: str) -> float:
    """Return supports_score − contradicts_score for the (claim, snippet) pair."""
    try:
        nli = _get_nli()
        result = nli(
            snippet[:512],
            candidate_labels=["supports this claim", "contradicts this claim", "unrelated"],
        )
        scores = dict(zip(result["labels"], result["scores"]))
        return scores.get("supports this claim", 0.0) - scores.get("contradicts this claim", 0.0)
    except Exception:
        return 0.0


class CriticAgent:
    def evaluate(self, claim: str) -> dict:
        # Gather sources
        sources = _ddg_api_sources(claim) + _ddg_html_sources(claim)
        top5 = sources[:5]

        # NLI score each source
        nli_scored = []
        for src in top5:
            score = _nli_score(claim, src["snippet"])
            nli_scored.append({**src, "nli_score": round(score, 4)})

        # Sort descending by NLI score
        nli_scored.sort(key=lambda x: x["nli_score"], reverse=True)

        # Confidence = average of top-3 NLI scores, clamped [0, 1]
        top3 = nli_scored[:3]
        avg_score = sum(s["nli_score"] for s in top3) / len(top3) if top3 else 0.0
        confidence = round(max(0.0, min(1.0, avg_score)), 2)

        # Verdict thresholds
        if avg_score >= 0.2:
            verdict: bool | None = True
        elif avg_score <= -0.1:
            verdict = False
        else:
            verdict = None

        # Correction: snippet with the most negative NLI score
        correction: str | None = None
        if verdict is False and nli_scored:
            correction = min(nli_scored, key=lambda x: x["nli_score"])["snippet"]

        clean_sources = [{"url": s["url"], "snippet": s["snippet"]} for s in nli_scored]
        nli_scores_out = [
            {"url": s["url"], "snippet": s["snippet"], "nli_score": s["nli_score"]}
            for s in nli_scored
        ]

        return {
            "verdict": verdict,
            "confidence": confidence,
            "sources": clean_sources,
            "nli_scores": nli_scores_out,
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
