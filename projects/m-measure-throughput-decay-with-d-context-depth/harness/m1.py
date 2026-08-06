import ref


def _rel_err(got, want):
    if want == 0.0:
        return 0.0 if got == 0.0 else 1.0
    return abs(got - want) / abs(want)


def check(workdir):
    try:
        from llamaperf.decay import measure_context_decay
    except Exception as e:
        return {"decay_rel_err": 1.0, "_note": f"Import failed: {e}"}

    max_err = 0.0
    for cfg, depths in zip(ref.CONFIGS, ref.DEPTH_SETS):
        try:
            want = ref.measure_context_decay(cfg, depths)
            got = measure_context_decay(cfg, depths)
        except Exception as e:
            return {"decay_rel_err": 1.0, "_note": f"Execution error: {e}"}

        if not isinstance(got, dict):
            return {"decay_rel_err": 1.0, "_note": "Returned value is not a dict"}

        err_kv = _rel_err(got.get("kv_bytes_per_token", 0.0), want["kv_bytes_per_token"])
        max_err = max(max_err, err_kv)

        got_tp = got.get("throughputs", [])
        want_tp = want["throughputs"]
        if len(got_tp) != len(want_tp):
            return {"decay_rel_err": 1.0, "_note": "throughputs length mismatch"}
        for g, w in zip(got_tp, want_tp):
            max_err = max(max_err, _rel_err(g, w))

        got_dec = got.get("decay_ratios", [])
        want_dec = want["decay_ratios"]
        if len(got_dec) != len(want_dec):
            return {"decay_rel_err": 1.0, "_note": "decay_ratios length mismatch"}
        for g, w in zip(got_dec, want_dec):
            max_err = max(max_err, _rel_err(g, w))

    return {"decay_rel_err": float(max_err)}
