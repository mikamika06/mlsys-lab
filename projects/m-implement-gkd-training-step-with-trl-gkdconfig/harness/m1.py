import ref


def check(workdir):
    from gkd.step import compute_gkd_step

    out = {"steps_matched": 0.0, "total": float(len(ref.TEST_CASES_STEP))}
    ok = 0
    for i, (s_logits, t_logits, cfg) in enumerate(ref.TEST_CASES_STEP):
        want = ref.compute_gkd_step(s_logits, t_logits, cfg)
        got = compute_gkd_step(s_logits, t_logits, cfg)
        if got is not None and abs(want["loss"] - got["loss"]) < 1e-4 and abs(want["grad_norm"] - got["grad_norm"]) < 1e-4:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"step case {i}: got {got}, reference {want}"
    out["steps_matched"] = float(ok)
    return out
