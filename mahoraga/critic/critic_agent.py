from typing import Optional


class CriticAgent:
    def evaluate(self, claim: str) -> dict:
        """Evaluate a claim and return a verdict with supporting metadata."""
        return {
            "verdict": False,
            "confidence": 0.0,
            "sources": [],
            "correction": None,
        }
