import sys
import os

def check(workdir):
    sys.path.insert(0, os.path.join(workdir))
    import ref
    from compat.adapter import transform_request
    
    out = {"shapes_matched": 0.0}
    ok = 0
    for shape in ref.SHAPES:
        raw = ref.make_sample_request(shape)
        try:
            res = transform_request(shape, raw)
            if isinstance(res, dict) and res.get("_shape") == shape and "adapter_version" in res:
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"Shape {shape} missing required _shape or adapter_version tags."
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"Failed on shape {shape}: {type(e).__name__}: {str(e)}"
    
    out["shapes_matched"] = float(ok)
    return out
