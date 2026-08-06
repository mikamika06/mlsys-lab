import ref


def check(workdir):
    from autoperf.profile import measure_speed_and_memory

    out = {"latency_ratio_valid": 0.0, "memory_measured": 0.0}
    model, x = ref.get_fixture()
    try:
        res = measure_speed_and_memory(model, x)
        if not isinstance(res, dict):
            out["_note"] = "measure_speed_and_memory must return a dictionary"
            return out

        for k in ["fp32_time", "bf16_time", "fp32_memory", "bf16_memory"]:
            if k not in res:
                out["_note"] = f"missing key {k} in result"
                return out

        if res["fp32_time"] > 0 and res["bf16_time"] >= 0:
            out["latency_ratio_valid"] = 1.0
        if res["fp32_memory"] > 0 and res["bf16_memory"] > 0:
            out["memory_measured"] = 1.0
    except Exception as e:
        out["_note"] = f"Execution failed: {type(e).__name__}: {str(e)[:120]}"
    return out
