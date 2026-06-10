from __future__ import annotations

import re
import sys
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

_nli_pipeline = None


# ─────────────────────────────────────────────────────────────
# Source collection
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
# Stage 1 — NLI fast gate
# ─────────────────────────────────────────────────────────────

def _get_nli():
    global _nli_pipeline
    if _nli_pipeline is None:
        from transformers import pipeline
        _nli_pipeline = pipeline(
            "zero-shot-classification",
            model="cross-encoder/nli-deberta-v3-small",
            token=False,  # bypass any stale cached HF credentials
        )
    return _nli_pipeline


def _nli_raw_probs(claim: str, snippet: str) -> tuple[float, float]:
    """Return (entail_prob, contra_prob) for the (claim, snippet) pair."""
    try:
        nli = _get_nli()
        result = nli(
            snippet[:512],
            candidate_labels=["supports this claim", "contradicts this claim", "unrelated"],
        )
        scores = dict(zip(result["labels"], result["scores"]))
        return (
            scores.get("supports this claim", 0.0),
            scores.get("contradicts this claim", 0.0),
        )
    except Exception:
        return (0.0, 0.0)


# ─────────────────────────────────────────────────────────────
# Stage 2 — Groq LLM judge
# ─────────────────────────────────────────────────────────────

def _claim_words(claim: str) -> set[str]:
    return {w.lower() for w in re.findall(r"\w+", claim)}


def _word_overlap(claim_words: set[str], text: str) -> float:
    if not text or not claim_words:
        return 0.0
    text_words = {w.lower() for w in re.findall(r"\w+", text)}
    return len(claim_words & text_words) / len(claim_words)


def _groq_judge(
    claim: str,
    top5: list[dict],
    nli_results: list[dict],
) -> tuple[bool | None, float, str | None]:
    """Call Groq LLM and return (verdict, confidence, correction)."""
    import os
    from dotenv import load_dotenv
    from groq import Groq

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")

    client = Groq(api_key=api_key)

    evidence = "\n".join(
        f"{i + 1}. {s['snippet'][:300]}"
        for i, s in enumerate(top5)
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise fact-checking assistant. You will be given a claim and up to 5 "
                    "web source snippets. Your job is to determine if the claim is factually correct "
                    "based on the evidence. Respond with EXACTLY one word: SUPPORTED, CONTRADICTED, or INSUFFICIENT."
                ),
            },
            {
                "role": "user",
                "content": f"CLAIM: {claim}\n\nEVIDENCE:\n{evidence}\n\nVerdict:",
            },
        ],
        max_tokens=10,
        temperature=0,
    )

    word = response.choices[0].message.content.strip().upper()

    cw = _claim_words(claim)
    overlaps = [_word_overlap(cw, s["snippet"]) for s in top5[:3]]
    avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0

    if word == "SUPPORTED":
        verdict: bool | None = True
        confidence = round((0.75 + avg_overlap) / 2, 2)
        correction = None
    elif word == "CONTRADICTED":
        verdict = False
        confidence = round((0.75 + avg_overlap) / 2, 2)
        most_contra = min(nli_results, key=lambda x: x["nli_score"]) if nli_results else None
        correction = most_contra["snippet"] if most_contra else (top5[0]["snippet"] if top5 else None)
    else:
        verdict = None
        confidence = 0.4
        correction = None

    return verdict, confidence, correction


def _overlap_fallback(
    claim: str,
    sources: list[dict],
) -> tuple[bool | None, float, str | None]:
    """Word-overlap scoring when Groq is unavailable."""
    cw = _claim_words(claim)
    scored = sorted(sources, key=lambda s: _word_overlap(cw, s["snippet"]), reverse=True)
    top3 = scored[:3]
    avg = sum(_word_overlap(cw, s["snippet"]) for s in top3) / len(top3) if top3 else 0.0
    confidence = round(min(1.0, avg), 2)

    if confidence >= 0.6:
        verdict: bool | None = True
    elif confidence < 0.3:
        verdict = False
    else:
        verdict = None

    correction = scored[0]["snippet"] if verdict is False and scored else None
    return verdict, confidence, correction


# ─────────────────────────────────────────────────────────────
# CriticAgent
# ─────────────────────────────────────────────────────────────

class CriticAgent:
    def evaluate(self, claim: str) -> dict:
        # Gather sources
        sources = _ddg_api_sources(claim) + _ddg_html_sources(claim)
        top5 = sources[:5]

        # ── Stage 1: NLI fast gate ────────────────────────────
        nli_results: list[dict] = []
        for src in top5:
            entail, contra = _nli_raw_probs(claim, src["snippet"])
            nli_results.append({
                **src,
                "entail_prob": entail,
                "contra_prob": contra,
                "nli_score": round(entail - contra, 4),
            })

        nli_results.sort(key=lambda x: x["nli_score"], reverse=True)

        clean_sources = [{"url": s["url"], "snippet": s["snippet"]} for s in nli_results]
        nli_scores_out = [
            {"url": s["url"], "snippet": s["snippet"], "nli_score": s["nli_score"]}
            for s in nli_results
        ]

        if nli_results:
            best_entail = max(nli_results, key=lambda x: x["entail_prob"])
            best_contra = max(nli_results, key=lambda x: x["contra_prob"])

            if best_entail["entail_prob"] >= 0.90:
                return {
                    "claim": claim,
                    "verdict": True,
                    "confidence": round(best_entail["entail_prob"], 2),
                    "correction": None,
                    "sources": clean_sources,
                    "nli_scores": nli_scores_out,
                    "scoring_method": "nli_fast",
                    "stage": 1,
                }

            if best_contra["contra_prob"] >= 0.90:
                return {
                    "claim": claim,
                    "verdict": False,
                    "confidence": round(best_contra["contra_prob"], 2),
                    "correction": best_contra["snippet"],
                    "sources": clean_sources,
                    "nli_scores": nli_scores_out,
                    "scoring_method": "nli_fast",
                    "stage": 1,
                }

        # ── Stage 2: Groq LLM judge ───────────────────────────
        try:
            verdict, confidence, correction = _groq_judge(claim, top5, nli_results)
            scoring_method = "groq_llm"
        except Exception as exc:
            print(f"[CRITIC] Groq unavailable ({exc}), using overlap fallback", file=sys.stderr)
            verdict, confidence, correction = _overlap_fallback(claim, nli_results)
            scoring_method = "fallback_overlap"

        return {
            "claim": claim,
            "verdict": verdict,
            "confidence": confidence,
            "correction": correction,
            "sources": clean_sources,
            "nli_scores": nli_scores_out,
            "scoring_method": scoring_method,
            "stage": 2,
        }


if __name__ == "__main__":
    import json

    agent = CriticAgent()
    for c in [
        "Python was created by Guido van Rossum",
        "The earth is flat",
    ]:
        print(f"\nClaim: {c}")
        print(json.dumps(agent.evaluate(c), indent=2))
