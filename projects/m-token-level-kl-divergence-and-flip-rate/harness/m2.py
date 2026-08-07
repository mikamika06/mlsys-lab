import sys
import os
import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"ppl_matched": 0.0, "recovery_matched": 0.0}

    try:
        from eval_metrics.perplexity import compute_perplexity_from_logprobs
        from eval_metrics.recovery import compute_recovery_percentage, parse_lm_eval_recovery
    except ImportError as e:
        out["_note"] = f"Import error: {e}"
        return out

    _, _, log_probs, b_eval, q_eval = ref.generate_test_data()

    want_ppl = ref.compute_perplexity_from_logprobs(log_probs)
    try:
        got_ppl = compute_perplexity_from_logprobs(log_probs)
        if abs(want_ppl - got_ppl) < 1e-4:
            out["ppl_matched"] = 1.0
        else:
            out["_note"] = f"Perplexity mismatch: want {want_ppl}, got {got_ppl}"
    except Exception as e:
        out["_note"] = f"Perplexity computation failed: {type(e).__name__}: {e}"
        return out

    want_rec = ref.parse_lm_eval_recovery(q_eval, b_eval, random_baseline=0.25)
    try:
        got_rec = parse_lm_eval_recovery(q_eval, b_eval, random_baseline=0.25)
        matched = True
        for k, want_v in want_rec.items():
            got_v = got_rec.get(k)
            if got_v is None or abs(want_v - got_v) > 1e-4:
                matched = False
                out["_note"] = f"Recovery mismatch for {k}: want {want_v}, got {got_v}"
                break
        if matched and len(got_rec) == len(want_rec):
            out["recovery_matched"] = 1.0
    except Exception as e:
        out["_note"] = f"Recovery calculation failed: {type(e).__name__}: {e}"
        return out

    return out
