from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()


class CriticAgent:
    def __init__(self):
        tavily_key = os.getenv("TAVILY_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")

        if not tavily_key:
            print("[CRITIC] Warning: TAVILY_API_KEY not set", file=sys.stderr)
        if not groq_key:
            print("[CRITIC] Warning: GROQ_API_KEY not set", file=sys.stderr)

        try:
            from tavily import TavilyClient
            self._tavily = TavilyClient(api_key=tavily_key) if tavily_key else None
        except Exception as e:
            print(f"[CRITIC] Tavily init failed: {e}", file=sys.stderr)
            self._tavily = None

        try:
            from groq import Groq
            self._groq = Groq(api_key=groq_key) if groq_key else None
        except Exception as e:
            print(f"[CRITIC] Groq init failed: {e}", file=sys.stderr)
            self._groq = None

    def evaluate(self, claim: str) -> dict:
        # ── Step A: Tavily search ─────────────────────────────
        sources: list[dict] = []
        direct_answer = ""
        scoring_method = "tavily+groq"

        try:
            if self._tavily is None:
                raise ValueError("Tavily client not initialized")
            result = self._tavily.search(
                query=claim,
                search_depth="basic",
                max_results=5,
                include_answer=True,
            )
            sources = [
                {
                    "url": s.get("url", ""),
                    "snippet": (s.get("content", "") or "")[:300],
                }
                for s in result.get("results", [])
            ]
            direct_answer = result.get("answer", "") or ""
        except Exception as e:
            print(f"[CRITIC] Tavily error: {e}", file=sys.stderr)
            scoring_method = "error_tavily"

        # ── Step B: Groq judgment ─────────────────────────────
        verdict: bool | None = None
        confidence = 0.0

        if scoring_method != "error_tavily":
            try:
                if self._groq is None:
                    raise ValueError("Groq client not initialized")

                evidence_parts = []
                if direct_answer:
                    evidence_parts.append(f"DIRECT ANSWER: {direct_answer}")
                for i, s in enumerate(sources, 1):
                    evidence_parts.append(f"{i}. {s['snippet']}")
                evidence = "\n".join(evidence_parts) if evidence_parts else "No evidence found."

                response = self._groq.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a precise fact-checking assistant. You will be given a claim "
                                "and web evidence. Determine if the claim is factually correct. "
                                "Respond with EXACTLY one word: SUPPORTED, CONTRADICTED, or INSUFFICIENT."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"CLAIM: {claim}\n\nEVIDENCE:\n{evidence}\n\nVerdict:",
                        },
                    ],
                    max_tokens=5,
                    temperature=0,
                )

                word = response.choices[0].message.content.strip().upper()

                if word == "SUPPORTED":
                    verdict = True
                    confidence = 0.82
                elif word == "CONTRADICTED":
                    verdict = False
                    confidence = 0.82
                else:
                    verdict = None
                    confidence = 0.40

            except Exception as e:
                print(f"[CRITIC] Groq error: {e}", file=sys.stderr)
                scoring_method = "error_groq"
                verdict = None
                confidence = 0.0

        # ── Step C: Correction ────────────────────────────────
        correction = None
        if verdict is False:
            correction = sources[0]["snippet"] if sources else (direct_answer or None)

        return {
            "claim": claim,
            "verdict": verdict,
            "confidence": confidence,
            "sources": sources[:3],
            "correction": correction,
            "scoring_method": scoring_method,
            "tavily_answer": direct_answer,
        }


if __name__ == "__main__":
    import json

    agent = CriticAgent()
    for claim in [
        "The sun is actually yellow",
        "Python was created by Guido van Rossum",
        "Dart does not support null safety",
        "1+1 equals 2",
        "The earth is flat",
    ]:
        print(f"\n{'─' * 60}\nClaim: {claim}")
        print(json.dumps(agent.evaluate(claim), indent=2))
