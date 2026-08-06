import ref


def _rel_err(got, want):
    if want == 0.0:
        return 0.0 if got == 0.0 else 1.0
    return abs(got - want) / abs(want)


def check(workdir):
    try:
        from llamaperf.offload import compare_offload_throughput
    except Exception as e:
        return {"offload_rel_err": 1.0, "_note": f"Import failed: {e}"}

    max_err = 0.0
    for test in ref.OFFLOAD_TESTS:
        cfg = test["config"]
        depth = test["depth"]
        ngl1 = test["ngl1"]
        ngl2 = test["ngl2"]
        try:
            want = ref.compare_offload_throughput(cfg, depth, ngl1, ngl2)
            got = compare_offload_throughput(cfg, depth, ngl1, ngl2)
        except Exception as e:
            return {"offload_rel_err": 1.0, "_note": f"Execution error: {e}"}

        if not isinstance(got, dict):
            return {"offload_rel_err": 1.0, "_note": "Returned value is not a dict"}

        for key in ("throughput_ngl1", "throughput_ngl2", "speedup", "offload_gain_tok_s"):
            if key not in got:
                return {"offload_rel_err": 1.0, "_note": f"Missing key: {key}"}
            err = _rel_err(float(got[key]), float(want[key]))
            max_err = max(max_err, err)

    return {"offload_rel_err": float(max_err)}
