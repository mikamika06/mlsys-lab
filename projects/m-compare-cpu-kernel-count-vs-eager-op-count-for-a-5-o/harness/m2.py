import ref


def check(workdir):
    out = {"config_matched": 0.0}
    try:
        from audit.core import find_autotuned_config
    except Exception as e:
        out["_note"] = f"import error: {e}"
        return out

    _, diff_cases = ref.generate_fixtures()
    ok = 0
    for i, case in enumerate(diff_cases):
        want = case["expected_config_id"]
        try:
            got = find_autotuned_config(case["target_diff"], case["candidates"])
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"case {i} raised {type(e).__name__}"
            continue
        if got == want:
            ok += 1
    out["config_matched"] = float(ok)
    return out
