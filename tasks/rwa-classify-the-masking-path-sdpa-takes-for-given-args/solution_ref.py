import numpy as np

def classify_masking(is_causal: bool,
                     attn_mask: np.ndarray | None) -> str:
    """
    Return a string describing which masking path SDPA will take.
    The implementation follows the documented rule set:
      * causal only → "causal"
      * explicit boolean mask → "bool_mask"
      * explicit numeric mask → "float_mask"
      * no mask → "none"
      * illegal combination (causal + explicit) → "illegal"
    """
    if is_causal:
        if attn_mask is None:
            return "causal"
        else:
            return "illegal"
    else:  # not causal
        if attn_mask is None:
            return "none"
        else:
            if isinstance(attn_mask, np.ndarray):
                if attn_mask.dtype.kind == 'b':
                    return "bool_mask"
                elif np.issubdtype(attn_mask.dtype, np.number):
                    return "float_mask"
    return "illegal"
