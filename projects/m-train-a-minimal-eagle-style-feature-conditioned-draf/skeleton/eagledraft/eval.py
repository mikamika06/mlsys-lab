import numpy as np


def evaluate_top1_accuracy(
    logits: np.ndarray, target_tokens: np.ndarray
) -> float:
    raise NotImplementedError


def compare_draft_heads(
    token_head, eagle_head, token_ids, hidden_states, target_tokens
):
    raise NotImplementedError
