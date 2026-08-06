import ref
import numpy as np

def check(workdir):
    from kvquant.extract import extract_scales
    ckpt = ref.generate_checkpoint()
    want = ref.extract_scales(ckpt)
    out = {"scales_matched": 0.0}
    try:
        got = extract_scales(ckpt)
        if isinstance(got, dict) and set(got.keys()) == set(want.keys()):
            match = True
            for k in want:
                if not np.allclose(got[k], want[k]):
                    match = False
                    break
            if match:
                out["scales_matched"] = 1.0
            else:
                out["_note"] = "extracted scale values do not match reference"
        else:
            out["_note"] = "extracted keys or return type mismatch"
    except Exception as e:
        out["_note"] = f"exception during extraction: {type(e).__name__}: {str(e)[:100]}"
    return out
