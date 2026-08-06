import ref
import torch

def check(workdir):
    from compengine.analyzer import analyze_latency
    from compengine.modes import select_mode

    out = {"profile_matched": 0.0, "ordering_valid": 0.0}
    model = ref.SimpleNet()
    inputs = ref.get_test_inputs()

    try:
        cfg = select_mode("default")
        res = analyze_latency(model, inputs, cfg)
        if isinstance(res, dict) and "compile_time" in res and "steady_latency" in res:
            out["profile_matched"] = 1.0
            if res["compile_time"] > 0 and res["steady_latency"] > 0:
                out["ordering_valid"] = 1.0
    except Exception as e:
        out["_note"] = f"Error during analysis: {str(e)[:100]}"
    return out
