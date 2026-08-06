import ref

def check(workdir):
    from cpuharness.modes import compare_cpu_modes
    out = {"modes_compared": 0.0, "latency_ratio_valid": 0.0}
    model = ref.get_test_model()
    inputs = ref.get_test_inputs()
    try:
        res = compare_cpu_modes(model, inputs)
        if isinstance(res, dict) and len(res) >= 4:
            out["modes_compared"] = 3.0
            if all(isinstance(v, float) for v in res.values()):
                out["latency_ratio_valid"] = 1.0
        else:
            out["_note"] = f"compare_cpu_modes returned invalid structure: {res}"
    except Exception as e:
        out["_note"] = f"compare_cpu_modes raised {type(e).__name__}: {str(e)[:100]}"
    return out
