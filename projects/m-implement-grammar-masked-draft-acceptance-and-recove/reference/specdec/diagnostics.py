import numpy as np


def compute_acceptance_loss(unmasked_runs, masked_runs):
    total_unmasked_accepted = sum(r["accepted_count"] for r in unmasked_runs)
    total_unmasked_drafted = sum(r["draft_count"] for r in unmasked_runs)

    total_masked_accepted = sum(r["accepted_count"] for r in masked_runs)
    total_masked_drafted = sum(r["draft_count"] for r in masked_runs)

    unmasked_rate = total_unmasked_accepted / max(1, total_unmasked_drafted)
    masked_rate = total_masked_accepted / max(1, total_masked_drafted)

    loss = max(0.0, unmasked_rate - masked_rate)

    return {
        "unmasked_rate": float(unmasked_rate),
        "masked_rate": float(masked_rate),
        "acceptance_loss": float(loss)
    }


def diagnose_run_collapse(run_a, run_b):
    drop_a = run_a["unmasked_rate"] - run_a["masked_rate"]
    drop_b = run_b["unmasked_rate"] - run_b["masked_rate"]

    if drop_a > drop_b and drop_a > 0.15:
        collapsed = "run_a"
        reason = "grammar_mismatch_collapse"
    elif drop_b > drop_a and drop_b > 0.15:
        collapsed = "run_b"
        reason = "grammar_mismatch_collapse"
    else:
        collapsed = "none"
        reason = "normal_variance"

    return {
        "collapsed_run": collapsed,
        "reason": reason,
        "max_drop": float(max(drop_a, drop_b))
    }
