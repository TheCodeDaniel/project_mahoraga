from mahoraga.trigger.trigger_evaluator import TriggerEvaluator


def test_should_trigger_returns_bool():
    evaluator = TriggerEvaluator()
    result = evaluator.should_trigger("some response", 0.5, False)
    assert isinstance(result, bool)


def test_should_trigger_with_user_flag():
    evaluator = TriggerEvaluator()
    result = evaluator.should_trigger("some response", 0.2, True)
    assert isinstance(result, bool)
