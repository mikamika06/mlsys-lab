import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from engine_diag.startup import parse_startup_log, total_startup_time
    except ImportError as e:
        return {"logs_parsed": 0.0, "_note": f"Import error: {e}"}

    out = {"logs_parsed": 0.0}
    ok = 0
    for i, sample in enumerate(ref.LOG_SAMPLES):
        want_phases = ref.parse_startup_log(sample)
        want_total = ref.total_startup_time(want_phases)

        try:
            got_phases = parse_startup_log(sample)
            got_total = total_startup_time(got_phases)
        except Exception as e:
            out["_note"] = f"Failed on log sample {i}: {e}"
            break

        if got_phases == want_phases and abs(got_total - want_total) < 1e-4:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"Sample {i}: got {got_phases}, want {want_phases}"

    out["logs_parsed"] = float(ok)
    return out
