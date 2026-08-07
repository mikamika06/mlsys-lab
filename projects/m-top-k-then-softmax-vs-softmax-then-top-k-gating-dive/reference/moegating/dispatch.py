import numpy as np


def build_mixtral_dispatch_tensor(
    selected_experts: np.ndarray, num_experts: int
) -> np.ndarray:
    num_tokens, top_k = selected_experts.shape
    dispatch = np.zeros((num_experts, num_tokens, top_k), dtype=np.int32)
    for t in range(num_tokens):
        for k_idx in range(top_k):
            exp_id = selected_experts[t, k_idx]
            dispatch[exp_id, t, k_idx] = 1
    return dispatch
