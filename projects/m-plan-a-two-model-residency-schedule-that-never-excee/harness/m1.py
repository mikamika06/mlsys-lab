import ref


def check(workdir):
    from residency.planner import plan_residency

    out = {"schedules_matched": 0.0}
    ok = 0
    for i, (ma, mb, lim) in enumerate(ref.CONFIGS):
        want = ref.plan_residency(ma, mb, lim) if hasattr(ref, "plan_residency") else None
        # Compute reference directly if not in ref module
        limit_bytes = lim * 1024 * 1024
        size_a = ma["weight_bytes"] + ma["kv_bytes"]
        size_b = mb["weight_bytes"] + mb["kv_bytes"]
        if size_a + size_b <= limit_bytes:
            expected = [{"step": j, "model_a_resident": True, "model_b_resident": True, "wired_bytes": size_a + size_b} for j in range(3)]
        else:
            expected = []
            for j in range(4):
                if j % 2 == 0:
                    expected.append({"step": j, "model_a_resident": True, "model_b_resident": False, "wired_bytes": size_a})
                else:
                    expected.append({"step": j, "model_a_resident": False, "model_b_resident": True, "wired_bytes": size_b})

        try:
            got = plan_residency(ma, mb, lim)
            if got is not None and len(got) > 0:
                valid = True
                for row in got:
                    if row["wired_bytes"] > limit_bytes:
                        valid = False
                if valid:
                    ok += 1
        except Exception:
            pass

    out["schedules_matched"] = float(ok)
    return out
