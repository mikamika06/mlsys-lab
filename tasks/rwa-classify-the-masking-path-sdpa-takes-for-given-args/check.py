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

def grade(sol, fx) -> dict:
    cases = [
        (True, None),
        (False, None),
        (False, [[True, False], [False, True]]),
        (False, [[0.0, -1e9], [-1e9, 0.0]]),
        (True, [[True, False], [False, True]]),
        (False, [1, 2, 3]),
        (False, [[True, False]])
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
