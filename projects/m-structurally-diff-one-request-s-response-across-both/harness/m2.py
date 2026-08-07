import ref


def check(workdir):
    from shimdiff.diff import find_ignored_parameter, recover_timings

    out = {"timings_matched": 0.0, "ignored_param_matched": 0.0}

    timings_ok = 0
    total_streams = len(ref.EVENT_STREAMS)
    for i in range(total_streams):
        want_t = ref.recover_timings(ref.EVENT_STREAMS[i])
        got_t = recover_timings(ref.EVENT_STREAMS[i])
        if got_t == want_t:
            timings_ok += 1
        elif "_note" not in out:
            out["_note"] = f"timing case {i}: got {got_t}, expected {want_t}"

    if timings_ok == total_streams:
        out["timings_matched"] = 1.0

    want_param = ref.find_ignored_parameter(
        ref.mock_runner, ref.BASE_PARAMS, ref.PARAM_CANDIDATES
    )
    got_param = find_ignored_parameter(
        ref.mock_runner, ref.BASE_PARAMS, ref.PARAM_CANDIDATES
    )

    if got_param == want_param:
        out["ignored_param_matched"] = 1.0
    elif "_note" not in out:
        out["_note"] = f"ignored param: got {got_param}, expected {want_param}"

    return out
