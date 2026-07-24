import numpy as np

def _oracle(is_causal, attn_mask):
    if is_causal:
        if attn_mask is None:
            return "causal"
        else:
            return "illegal"
    else:
        if attn_mask is None:
            return "none"
        else:
            if isinstance(attn_mask, np.ndarray):
                if attn_mask.dtype.kind == 'b':
                    return "bool_mask"
                elif np.issubdtype(attn_mask.dtype, np.number):
                    return "float_mask"
    return "illegal"

def grade(sol, fx) -> dict:
    cases = [
        (True, None),
        (False, None),
        (False, np.array([[True, False], [False, True]])),
        (False, np.array([[0.0, -1e9], [-1e9, 0.0]], dtype=np.float32)),
        (True, np.array([[True, False], [False, True]])),
        (False, np.array([1, 2, 3], dtype=int)),
        (False, np.array([[True, False]], dtype=bool))
    ]
    ok = 1.0
    for is_causal, mask in cases:
        try:
            got = sol.classify_masking(is_causal, mask)
        except Exception:
            return {"exact_match": 0.0}
        expected = _oracle(is_causal, mask)
        if got != expected:
            return {"exact_match": 0.0}
    return {"exact_match": ok}
