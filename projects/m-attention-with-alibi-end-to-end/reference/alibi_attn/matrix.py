SUPPORT_MATRIX = {
    "standard": {"alibi": True, "softcap": True, "causal": True, "sliding_window": True},
    "flash_attn": {"alibi": True, "softcap": False, "causal": True, "sliding_window": True},
    "paged_attn": {"alibi": True, "softcap": True, "causal": True, "sliding_window": False},
}

def check_support_matrix(backend, modifiers):
    if backend not in SUPPORT_MATRIX:
        return False
    return all(SUPPORT_MATRIX[backend].get(m, False) for m in modifiers)
