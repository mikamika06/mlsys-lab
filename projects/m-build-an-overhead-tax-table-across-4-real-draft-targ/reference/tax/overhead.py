from tax.profiles import calculate_expected_acceptance


def compute_pair_tax(pair_data: dict, gamma: int) -> dict:
    """Compute overhead tax and speedup for a draft/target pair at depth gamma."""
    acc_info = calculate_expected_acceptance(pair_data["acceptance_probs"], gamma)
    expected_accepted = acc_info["expected_accepted"]

    draft_step_ms = pair_data["draft_step_ms"]
    target_step_ms = pair_data["target_step_ms"]
    verify_step_ms = pair_data["verify_step_ms"][gamma]

    total_draft_ms = draft_step_ms * gamma
    total_step_ms = total_draft_ms + verify_step_ms

    effective_latency_ms = total_step_ms / expected_accepted
    speedup = target_step_ms / effective_latency_ms
    overhead_tax = (effective_latency_ms - target_step_ms) / target_step_ms

    return {
        "pair_id": pair_data["pair_id"],
        "gamma": gamma,
        "total_step_ms": total_step_ms,
        "expected_accepted": expected_accepted,
        "effective_latency_ms": effective_latency_ms,
        "speedup": speedup,
        "overhead_tax": overhead_tax,
    }
