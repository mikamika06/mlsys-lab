import ref


def check(workdir):
    out = {"sim_match_no_age": 0.0, "sim_match_age": 0.0}

    try:
        from vllm_policy.scheduler import simulate
    except Exception as e:
        out["_note"] = f"import failed: {e}"
        return out

    s_ok_0 = 0
    s_ok_1 = 0
    for reqs in ref.REQ_CASES:
        try:
            want_0 = ref.simulate(reqs, 0.0)
            got_0 = simulate(reqs, 0.0)
            if got_0 == want_0:
                s_ok_0 += 1
        except Exception:
            pass

        try:
            want_1 = ref.simulate(reqs, 1.0)
            got_1 = simulate(reqs, 1.0)
            if got_1 == want_1:
                s_ok_1 += 1
        except Exception:
            pass

    if ref.REQ_CASES:
        out["sim_match_no_age"] = float(s_ok_0) / len(ref.REQ_CASES)
        out["sim_match_age"] = float(s_ok_1) / len(ref.REQ_CASES)

    return out
