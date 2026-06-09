class TriggerEvaluator:
    def should_trigger(self, response: str, confidence: float, user_flagged: bool) -> bool:
        """Decide whether the adaptation pipeline should be triggered."""
        return False
