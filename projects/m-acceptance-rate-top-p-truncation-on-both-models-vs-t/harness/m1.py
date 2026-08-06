import numpy as np
import ref


def check(workdir):
    from spec.sampling import apply_temperature_and_topp, compute_acceptance_prob, sample_residual

    cases = ref.generate_test_cases()
    matched = 0
    out = {"acceptance_matches": 0.0, "total": float(len(cases))}
    for i, c in enumerate(cases):
        ref_prob = compute_acceptance_prob(
            c["p_logits"], c["q_logits"], c["token_id"], c["temp"], c["p_target"], c["p_draft"]
        )
        got_prob = compute_acceptance_prob(
            c["p_logits"], c["q_logits"], c["token_id"], c["temp"], c["p_target"], c["p_draft"]
        )
        if np.isclose(ref_prob, got_prob, atol=1e-5):
            matched += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: expected prob {ref_prob:.4f}, got {got_prob:.4f}"
    out["acceptance_matches"] = float(matched)
    return out
