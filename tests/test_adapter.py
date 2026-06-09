from mahoraga.adapter.adapter_manager import AdapterManager


def test_update_returns_bool():
    manager = AdapterManager()
    correction = {"verdict": True, "confidence": 0.9, "sources": [], "correction": "Fix text"}
    result = manager.update(correction)
    assert isinstance(result, bool)


def test_update_with_empty_correction():
    manager = AdapterManager()
    result = manager.update({})
    assert isinstance(result, bool)
