import ref
import numpy as np


def check(workdir):
    from kvcache.mask import reconstruct_kept_mask

    scenarios = ref.generate_mask_scenarios()
    matched = 0
    for sc in scenarios:
        try:
            mask = reconstruct_kept_mask(sc["dump"], sc["length"])
            expected = ref.reconstruct_kept_mask(sc["dump"], sc["length"])
            if isinstance(mask, np.ndarray) and mask.dtype == bool and np.array_equal(mask, expected):
                matched += 1
        except Exception:
            pass
    return {"masks_matched": float(matched)}
