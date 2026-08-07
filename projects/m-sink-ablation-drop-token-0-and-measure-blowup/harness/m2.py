import ref

def check(workdir):
    import numpy as np

    out = {"ablation_correct": 0.0, "blowup_correct": 0.0}
    try:
        from sink_ablate.ablation import drop_token_0, measure_blowup
        want_ab = ref.drop_token_0(ref.PROBS)
        got_ab = drop_token_0(ref.PROBS)

        if got_ab is not None and np.allclose(want_ab, got_ab, atol=1e-6):
            out["ablation_correct"] = 1.0

        want_bl = ref.measure_blowup(ref.PROBS, want_ab)
        got_bl = measure_blowup(ref.PROBS, got_ab)

        if got_bl is not None and abs(want_bl - got_bl) < 1e-6:
            out["blowup_correct"] = 1.0
    except Exception as e:
        out["_note"] = f"{type(e).__name__}: {str(e)}"
    return out
