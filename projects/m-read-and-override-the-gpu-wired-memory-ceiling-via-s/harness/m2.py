import sys
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    out = {"log_diagnoses_matched": 0.0}

    try:
        from sysctl_mem.diagnose import parse_server_log
    except Exception as e:
        out["_note"] = f"Failed to import diagnose functions: {e}"
        return out

    ok = True
    for idx, log in enumerate(ref.SAMPLE_LOGS):
        want = ref.parse_server_log(log)
        try:
            got = parse_server_log(log)
            if got != want:
                ok = False
                out["_note"] = f"Sample log {idx}: got {got}, want {want}"
                break
        except Exception as e:
            ok = False
            out["_note"] = f"parse_server_log raised exception on sample log {idx}: {e}"
            break

    if ok:
        out["log_diagnoses_matched"] = 1.0

    return out
