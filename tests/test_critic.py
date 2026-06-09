from mahoraga.critic.critic_agent import CriticAgent


def test_evaluate_returns_expected_keys():
    agent = CriticAgent()
    result = agent.evaluate("Some claim")
    assert "verdict" in result
    assert "confidence" in result
    assert "sources" in result
    assert "correction" in result


def test_evaluate_verdict_is_bool():
    agent = CriticAgent()
    result = agent.evaluate("Some claim")
    assert isinstance(result["verdict"], bool)


def test_evaluate_confidence_is_float():
    agent = CriticAgent()
    result = agent.evaluate("Some claim")
    assert isinstance(result["confidence"], float)


def test_evaluate_sources_is_list():
    agent = CriticAgent()
    result = agent.evaluate("Some claim")
    assert isinstance(result["sources"], list)
