import numpy as np


def check(workdir):
    from spec.acceptance import evaluate_acceptance_rate

    out = {"rate_analysis_correct": 0.0}
    np.random.seed(2026)
    seq_len = 20
    vocab_size = 16
    t_logits = np.random.randn(seq_len, vocab_size)
    d_logits = np.random.randn(seq_len, vocab_size)
    d_tokens = np.random.randint(0, vocab_size, size=seq_len)

    res_dual = evaluate_acceptance_rate(t_logits, d_logits, d_tokens, temperature=0.8, top_p_target=0.8, top_p_draft=0.8)
    res_target_only = evaluate_acceptance_rate(t_logits, d_logits, d_tokens, temperature=0.8, top_p_target=0.8, top_p_draft=1.0)

    if not isinstance(res_dual, dict) or not isinstance(res_target_only, dict):
        out["_note"] = "evaluate_acceptance_rate did not return a dictionary"
        return out

    keys = ["empirical_acceptance_rate", "expected_acceptance_rate", "per_token_prob"]
    for k in keys:
        if k not in res_dual or k not in res_target_only:
            out["_note"] = f"missing key {k} in returned dict"
            return out

    if len(res_dual["per_token_prob"]) != seq_len:
        out["_note"] = f"per_token_prob length mismatch: got {len(res_dual['per_token_prob'])}, expected {seq_len}"
        return out

    if res_target_only["expected_acceptance_rate"] >= res_dual["expected_acceptance_rate"] - 1e-6:
        out["rate_analysis_correct"] = 1.0
    else:
        out["_note"] = f"expected target-only acceptance ({res_target_only['expected_acceptance_rate']:.4f}) to be >= dual top-p acceptance ({res_dual['expected_acceptance_rate']:.4f})"

    return out
