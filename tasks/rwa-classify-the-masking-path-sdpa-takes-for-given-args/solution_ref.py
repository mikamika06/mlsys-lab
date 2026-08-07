def classify_masking(is_causal: bool,
                     attn_mask: list | None) -> str:
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
            if isinstance(attn_mask, list):
                elements = []
                stack = [attn_mask]
                while stack:
                    curr = stack.pop()
                    if isinstance(curr, list):
                        stack.extend(curr)
                    else:
                        elements.append(curr)

                if not elements:
                    return "illegal"

                all_bool = True
                all_num = True
                for el in elements:
                    if isinstance(el, bool):
                        all_num = False
                    elif isinstance(el, (int, float)):
                        all_bool = False
                    else:
                        all_bool = False
                        all_num = False

                if all_bool:
                    return "bool_mask"
                elif all_num:
                    return "float_mask"
    return "illegal"
