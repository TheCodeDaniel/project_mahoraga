import json
from datetime import datetime, timezone
from pathlib import Path

_KB_PATH = Path(__file__).parent / "knowledge_base.json"


class AdapterManager:
    def update(self, correction: dict) -> dict:
        """Persist a correction to the local knowledge base."""
        claim = correction.get("claim", "")

        if correction.get("verdict") is True or correction.get("correction") is None:
            return {"status": "skipped", "reason": "no_correction_needed", "claim": claim}

        entry = {
            "claim": claim,
            "correction": correction["correction"],
            "sources": correction.get("sources", [])[:3],
            "confidence": correction.get("confidence", 0.0),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "applied": True,
        }

        entries = self.read_knowledge_base()
        entries.append(entry)
        _KB_PATH.write_text(json.dumps(entries, indent=2))

        print(f"[ADAPTER] Knowledge base updated — claim: {claim[:60]}...")
        return {"status": "updated", "entry": entry}

    def read_knowledge_base(self) -> list:
        """Return all entries from knowledge_base.json, or [] if it doesn't exist."""
        if not _KB_PATH.exists():
            return []
        try:
            return json.loads(_KB_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return []
