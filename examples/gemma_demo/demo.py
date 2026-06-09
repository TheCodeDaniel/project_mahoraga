from mahoraga.critic.critic_agent import CriticAgent
from mahoraga.trigger.trigger_evaluator import TriggerEvaluator
from mahoraga.adapter.adapter_manager import AdapterManager
from mahoraga.orchestrator.maestro_hook import MaestroHook


def main():
    critic = CriticAgent()
    trigger = TriggerEvaluator()
    adapter = AdapterManager()
    hook = MaestroHook()

    claim = "The Eiffel Tower is located in Berlin."
    print(f"[Demo] Evaluating claim: '{claim}'")

    result = critic.evaluate(claim)
    print(f"[Critic] verdict={result['verdict']}, confidence={result['confidence']}, correction={result['correction']}")

    should_adapt = trigger.should_trigger(
        response=claim,
        confidence=result["confidence"],
        user_flagged=True,
    )
    print(f"[Trigger] should trigger adaptation: {should_adapt}")

    if should_adapt:
        case_id = hook.open_case({"claim": claim, "critic_result": result})
        print(f"[Orchestrator] opened case: {case_id}")

        updated = adapter.update(result)
        print(f"[Adapter] update applied: {updated}")
    else:
        print("[Pipeline] No adaptation triggered — confidence threshold not met or not flagged.")


if __name__ == "__main__":
    main()
