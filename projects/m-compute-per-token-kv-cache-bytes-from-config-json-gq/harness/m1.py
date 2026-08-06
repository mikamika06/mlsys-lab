import sys
import ref


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    out = {"configs_matched": 0.0}
    try:
        from vllm_budget.kv import bytes_per_token
    except Exception as e:
        out["_note"] = f"Failed to import bytes_per_token: {e}"
        return out

    ok = 0
    total = len(ref.CONFIGS_FIXTURES)
    for fix in ref.CONFIGS_FIXTURES:
        want = ref.ref_bytes_per_token(fix["config"], fix["dtype"])
        try:
            got = bytes_per_token(fix["config"], fix["dtype"])
            if got == want:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"Expected {want} bytes/token, got {got}"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"Error during bytes_per_token execution: {e}"

    if ok == total:
        out["configs_matched"] = 1.0
    return out
