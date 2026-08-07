import sys
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    out = {"prediction_accuracy": 0.0, "override_validity": 0.0}

    try:
        from sysctl_mem.ceiling import predict_wired_limit_mb, generate_sysctl_override
    except Exception as e:
        out["_note"] = f"Failed to import ceiling functions: {e}"
        return out

    pred_ok = True
    for mem in ref.SAMPLE_MEMSIZES:
        want = ref.predict_wired_limit_mb(mem)
        try:
            got = predict_wired_limit_mb(mem)
            if got != want:
                pred_ok = False
                out["_note"] = f"predict_wired_limit_mb({mem}): got {got}, want {want}"
                break
        except Exception as e:
            pred_ok = False
            out["_note"] = f"predict_wired_limit_mb({mem}) raised exception: {e}"
            break

    if pred_ok:
        out["prediction_accuracy"] = 1.0

    override_ok = True
    for mem in ref.SAMPLE_MEMSIZES:
        for pct in [60.0, 75.0, 90.0]:
            want_cmd = ref.generate_sysctl_override(mem, pct)
            try:
                got_cmd = generate_sysctl_override(mem, pct)
                if got_cmd != want_cmd:
                    override_ok = False
                    out["_note"] = f"generate_sysctl_override({mem}, {pct}): got '{got_cmd}', want '{want_cmd}'"
                    break
            except Exception as e:
                override_ok = False
                out["_note"] = f"generate_sysctl_override({mem}, {pct}) raised exception: {e}"
                break
        if not override_ok:
            break

    try:
        generate_sysctl_override(64 * 1024 * 1024 * 1024, 99.0)
        override_ok = False
        out["_note"] = "generate_sysctl_override failed to raise ValueError for percentage > 95"
    except ValueError:
        pass
    except Exception as e:
        override_ok = False
        out["_note"] = f"generate_sysctl_override raised wrong exception type: {type(e).__name__}"

    if override_ok:
        out["override_validity"] = 1.0

    return out
