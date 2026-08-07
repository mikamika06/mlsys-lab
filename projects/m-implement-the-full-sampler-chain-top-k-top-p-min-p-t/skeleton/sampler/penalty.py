import numpy as np


def apply_repeat_penalty(logits, input_ids, penalty=1.0, repeat_last_n=64):
    """Applies repeat penalty to logits for tokens within repeat_last_n window."""
    raise NotImplementedError
