import uuid


class MaestroHook:
    def open_case(self, trigger_data: dict) -> str:
        """Open a new case for investigation and return a unique case ID."""
        return f"case-{uuid.uuid4().hex[:8]}"
