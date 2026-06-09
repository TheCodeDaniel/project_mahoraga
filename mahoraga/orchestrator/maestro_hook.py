import uuid
from datetime import datetime, timezone


class MaestroHook:
    def __init__(self):
        self._cases: dict[str, dict] = {}

    def open_case(self, trigger_data: dict) -> dict:
        """Open a new case and store it locally (UiPath integration pending Lab access)."""
        case_id = str(uuid.uuid4())
        case = {
            "case_id": case_id,
            "status": "open",
            "trigger_data": trigger_data,
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "resolution": None,
        }
        self._cases[case_id] = case
        print(f"[MAESTRO] Case {case_id} opened — reason: {trigger_data.get('reason')}")
        return case

    def close_case(self, case_id: str, resolution: str) -> dict:
        """Close an existing case with a resolution string."""
        case = self._cases[case_id]
        case["status"] = "closed"
        case["resolution"] = resolution
        case["closed_at"] = datetime.now(timezone.utc).isoformat()
        print(f"[MAESTRO] Case {case_id} closed — resolution: {resolution}")
        return case
