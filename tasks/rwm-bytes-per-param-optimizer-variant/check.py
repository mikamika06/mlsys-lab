import numpy as np

def _ref(state_dict):
    total_bytes = sum(arr.nbytes for arr in state_dict.values())
    num_params = next(iter(state_dict.values())).size
    bpp = total_bytes / num_params
    if abs(bpp - 8) < 0.5:
        return "adam_fp32"
    elif abs(bpp - 4) < 0.5:
        return "adam_fp16"
    else:
        return "adam_uint8"

def grade(sol, fx) -> dict:
    cases = []
    n = 10
    state_fp32 = {"m": np.arange(n, dtype=np.float32),
                  "v": np.arange(n, dtype=np.float32)}
    cases.append(state_fp32)
    state_fp16 = {"m": np.arange(n, dtype=np.float16),
                  "v": np.arange(n, dtype=np.float16)}
    cases.append(state_fp16)
    state_uint8 = {"m": np.arange(n, dtype=np.uint8),
                   "v": np.arange(n, dtype=np.uint8)}
    cases.append(state_uint8)

    ok = 1.0
    for state in cases:
        try:
            got = sol.optimizer_variant(state)
        except Exception:
            return {"exact_match": 0.0}
        ref = _ref(state)
        if got != ref:
            return {"exact_match": 0.0}
    return {"exact_match": ok}
