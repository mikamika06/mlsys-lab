import numpy as np


def check_tied(state_dict):
    """Check if embed tokens and lm head are tied."""
    w_embed = state_dict.get("model.embed_tokens.weight")
    w_head = state_dict.get("lm_head.weight")
    if w_embed is None or w_head is None:
        return False
    if w_embed is w_head:
        return True
    return np.array_equal(w_embed, w_head) or np.shares_memory(w_embed, w_head)
