import json

from mahoraga.adapter.adapter_manager import AdapterManager
from mahoraga.critic.critic_agent import CriticAgent
from mahoraga.orchestrator.maestro_hook import MaestroHook
from mahoraga.trigger.trigger_evaluator import TriggerEvaluator


def run_pipeline(claim: str, ai_response: str, user_flagged: bool = False) -> dict:
    print("\n=== MAHORAGA PIPELINE ===")
    print(f"Claim     : {claim}")
    print(f"AI Response: {ai_response}")

    # Step 1 — Critic
    critic_result = CriticAgent().evaluate(claim)
    print(f"\n[CRITIC] verdict={critic_result['verdict']}  confidence={critic_result['confidence']}")

    # Step 2 — Trigger
    trigger = TriggerEvaluator().should_trigger(
        response=ai_response,
        confidence=critic_result["confidence"],
        user_flagged=user_flagged,
    )
    print(
        f"[TRIGGER] triggered={trigger['triggered']}  reason={trigger['reason']}"
        f"  self_confidence={trigger['self_confidence']}"
    )

    case_id: str | None = None
    hook = MaestroHook()

    if trigger["triggered"]:
        # Step 3 — Open case
        case = hook.open_case(trigger)
        case_id = case["case_id"]
        print(f"\n[ORCHESTRATOR] Case opened:")
        print(json.dumps({k: v for k, v in case.items() if k != "trigger_data"}, indent=2))

        # Step 4 — Human review simulation
        if critic_result.get("correction"):
            print(f"\n[HUMAN REVIEW] Correction approved: {critic_result['correction'][:120]}")
            resolution = "adapter_updated"
        else:
            print("\n[HUMAN REVIEW] No correction needed")
            resolution = "no_update_needed"

        # Step 5 — Adapter
        updated = AdapterManager().update(critic_result)
        print(f"[ADAPTER] update applied: {updated}")

        # Step 6 — Close case
        hook.close_case(case_id, resolution)
    else:
        print("\n[MAHORAGA] No trigger. Response confidence acceptable.")

    summary = {
        "claim": claim,
        "verdict": critic_result["verdict"],
        "confidence": critic_result["confidence"],
        "triggered": trigger["triggered"],
        "case_id": case_id,
    }
    print("\n--- Summary ---")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_pipeline(
        claim="flutter_gemma supports iOS 12",
        ai_response="I think flutter_gemma supports iOS 12 or higher",
        user_flagged=False,
    )

    run_pipeline(
        claim="Python was created in 1991 by Guido van Rossum",
        ai_response="Python was definitely created in 1991 by Guido van Rossum",
        user_flagged=False,
    )
