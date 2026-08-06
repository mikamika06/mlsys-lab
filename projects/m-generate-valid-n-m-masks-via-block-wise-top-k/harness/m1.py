import ref
import numpy as np


def check(workdir):
    from sparse.masks import generate_nm_mask
    out = {"masks_valid": 0.0}
    test_tensor = np.random.randn(8, 16)
    try:
        mask = generate_nm_mask(test_tensor, n=2, m=4)
        mask_arr = np.array(mask)
        if mask_arr.shape != test_tensor.shape:
            out["_note"] = f"shape mismatch: {mask_arr.shape} vs {test_tensor.shape}"
            return out
        reshaped = mask_arr.reshape(-1, 4)
        sums = np.sum(reshaped, axis=1)
        if not np.all(sums == 2):
            out["_note"] = f"block sums are not all 2: {sums}"
            return out
        if not np.all((mask_arr == 0) | (mask_arr == 1)):
            out["_note"] = "mask contains values other than 0 and 1"
            return out
        out["masks_valid"] = 1.0
    except Exception as e:
        out["_note"] = f"exception raised: {type(e).__name__}: {str(e)[:120]}"
    return out
