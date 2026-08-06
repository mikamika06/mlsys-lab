import ref
import numpy as np


def check(workdir):
    from nmvalidate.masks import extract_nm_mask
    out = {"masks_matched": 0.0, "total": 0.0}
    valid_cases = [tc for tc in ref.TEST_CASES if tc["valid"]]
    out["total"] = float(len(valid_cases))
    ok = 0
    for i, tc in enumerate(valid_cases):
        try:
            mask = extract_nm_mask(tc["weight"], tc["n"], tc["m"], tc["dim"])
            expected_mask = (tc["weight"] != 0).astype(np.uint8)
            if np.array_equal(mask, expected_mask):
                ok += 1
            elif "_note" not in out:
                out["_note"] = f"mask case {i} mismatch"
        except Exception as e:
            if "_note" not in out:
                out["_note"] = f"mask case {i} raised {type(e).__name__}"
    out["masks_matched"] = float(ok)
    return out
