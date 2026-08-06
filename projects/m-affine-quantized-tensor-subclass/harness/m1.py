import sys


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    out = {"configs_matched": 0.0, "has_attributes": 0.0}

    try:
        from qtensor.configs import map_target_to_config
        targets = ["edge_device", "fine_tuning", "server_inference"]
        cfgs = [map_target_to_config(t) for t in targets]
        if all(isinstance(c, dict) for c in cfgs):
            if all(k in cfgs[0] for k in ["method", "group_size", "asymmetric"]):
                out["configs_matched"] = 3.0
    except Exception as e:
        out["_note_m1_configs"] = str(e)

    try:
        import numpy as np
        from qtensor.subclass import quantize_affine
        w = np.random.randn(32, 32).astype(np.float32)
        q = quantize_affine(w, 16, True)
        if hasattr(q, "int_data") and hasattr(q, "scale") and hasattr(q, "zero_point"):
            out["has_attributes"] = 1.0
    except Exception as e:
        out["_note_m1_attr"] = str(e)

    return out
