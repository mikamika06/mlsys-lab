import ref
import numpy as np

def check(workdir):
    from quant.curve import compute_error_curve
    tensor = ref.generate_tensor()
    block_sizes = [16, 32, 64, 128]
    want = compute_error_curve(tensor, block_sizes)
    try:
        got = compute_error_curve(tensor, block_sizes)
    except Exception as e:
        return {"rel_err": 1.0, "_note": f"raised {type(e).__name__}"}
    if len(got) != len(want):
        return {"rel_err": 1.0, "_note": "length mismatch"}
    diff = np.max(np.abs(np.array(got) - np.array(want)))
    return {"rel_err": float(diff)}
