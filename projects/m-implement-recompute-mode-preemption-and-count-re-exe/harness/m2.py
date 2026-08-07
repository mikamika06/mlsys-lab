import ref


def check(workdir):
    from preempt.swap import compute_swap_cost
    from preempt.selector import choose_preemption_mode

    out = {"swap_rel_err": 1.0, "profiles_matched": 0}

    ref_cost = ref.compute_swap_cost(128, 1024 * 1024, 16.0, roundtrip=True)
    try:
        got_cost = compute_swap_cost(128, 1024 * 1024, 16.0, roundtrip=True)
        time_err = abs(got_cost["time_seconds"] - ref_cost["time_seconds"]) / ref_cost["time_seconds"]
        bytes_err = abs(got_cost["bytes_moved"] - ref_cost["bytes_moved"]) / ref_cost["bytes_moved"]
        out["swap_rel_err"] = float(max(time_err, bytes_err))
    except Exception as e:
        out["_note"] = f"compute_swap_cost error: {e}"
        return out

    matched_count = 0
    for profile in ref.WORKLOAD_PROFILES:
        want_mode = ref.choose_preemption_mode(profile)
        try:
            got_mode = choose_preemption_mode(profile)
            if got_mode == want_mode:
                matched_count += 1
        except Exception as e:
            out["_note"] = f"choose_preemption_mode error: {e}"
            break

    out["profiles_matched"] = matched_count
    return out
