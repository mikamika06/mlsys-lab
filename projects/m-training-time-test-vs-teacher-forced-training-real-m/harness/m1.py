import ref


def check(workdir):
    from spectrain.teacher import teacher_forced_loss

    out = {"teacher_vs_rollout_matched": 0.0, "configs": float(len(ref.CONFIGS))}
    ok = 0
    for i, cfg in enumerate(ref.CONFIGS):
        want = ref.compute_teacher_forced_loss(cfg["tokens"], cfg["draft_logits"])
        try:
            got = teacher_forced_loss(cfg["tokens"], cfg["draft_logits"])
        except Exception:
            got = -1.0
        if abs(got - want) < 1e-4:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"config {i}: got {got}, reference {want}"
    out["teacher_vs_rollout_matched"] = float(ok)
    return out
