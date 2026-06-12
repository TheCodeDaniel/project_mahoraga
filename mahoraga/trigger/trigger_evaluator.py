import re

_HEDGE_PATTERN = re.compile(
    r"\b(i think|i believe|not sure|might|probably|possibly|unclear|i'm not certain|may be)\b",
    re.IGNORECASE,
)
_ASSERT_PATTERN = re.compile(
    r"\b(definitely|certainly|always|never|guaranteed|absolutely|confirmed)\b",
    re.IGNORECASE,
)


class TriggerEvaluator:
    def should_trigger(
        self,
        response: str,
        confidence: float,
        user_flagged: bool,
        critic_verdict: bool | None = None,
    ) -> dict:
        if _HEDGE_PATTERN.search(response):
            self_confidence = 0.35
        elif _ASSERT_PATTERN.search(response):
            self_confidence = 0.85
        else:
            self_confidence = 0.6

        effective_confidence = confidence if confidence is not None else 0.0

        if user_flagged:
            triggered, reason = True, "user_flagged"
        elif critic_verdict is False:
            triggered, reason = True, "critic_disagreement"
        elif effective_confidence < 0.35:
            triggered, reason = True, "low_confidence"
        elif self_confidence <= 0.35:
            triggered, reason = True, "hedged_response"
        else:
            triggered, reason = False, None

        return {
            "triggered": triggered,
            "reason": reason,
            "self_confidence": self_confidence,
            "confidence": effective_confidence,
            "response_preview": response[:120],
        }
