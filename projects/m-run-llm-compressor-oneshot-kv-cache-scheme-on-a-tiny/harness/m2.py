import ref
import numpy as np

def check(workdir):
    from compressor_kv.debug import repair_recipe
    out = {"recipe_repaired": 0.0, "max_scale_valid": 0.0}
    cfg = ref.CONFIGS[0]
    acts = ref.generate_calibration_data(cfg)
    broken_recipe = {"scale": 1.0, "scheme": "fp8_e4m3"}
    try:
        repaired = repair_recipe(broken_recipe, acts)
    except Exception as e:
        out["_note"] = f"repair raised: {type(e).__name__}: {str(e)[:100]}"
        return out

    if isinstance(repaired, dict) and repaired.get("scale") != 1.0:
        out["recipe_repaired"] = 1.0
        if 0.0 < repaired.get("scale", 0.0) < 10.0:
            out["max_scale_valid"] = 1.0
        else:
            out["_note"] = f"repaired scale out of expected bounds: {repaired.get('scale')}"
    else:
        out["_note"] = f"recipe not repaired, got {repaired}"
    return out
