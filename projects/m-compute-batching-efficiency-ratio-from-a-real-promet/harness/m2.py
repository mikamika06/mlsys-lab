import ref


def check(workdir):
    from batching.latency import decompose_latency
    out = {"rel_err": 1.0}
    dump = ref.generate_fixture(seed=202)
    want = ref.decompose_latency(dump)
    try:
        got = decompose_latency(dump)
    except Exception as e:
        out["_note"] = f"raised exception: {e}"
        return out

    if not isinstance(got, dict) or "queue_fraction" not in got or "exec_fraction" not in got:
        out["_note"] = f"invalid return format: {got}"
        return out

    err_q = abs(got["queue_fraction"] - want["queue_fraction"]) / (abs(want["queue_fraction"]) + 1e-9)
    err_e = abs(got["exec_fraction"] - want["exec_fraction"]) / (abs(want["exec_fraction"]) + 1e-9)
    out["rel_err"] = float(max(err_q, err_e))
    return out
