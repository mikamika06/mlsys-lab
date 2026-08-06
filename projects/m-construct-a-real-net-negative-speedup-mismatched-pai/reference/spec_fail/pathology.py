import numpy as np

def get_net_negative_speedup_config() -> tuple[np.ndarray, np.ndarray, int, float, float]:
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    gamma = 4
    t_draft = 20.0
    t_target = 50.0
    return p, q, gamma, t_draft, t_target

def prompt_lookup_draft(sequence: list[int], gamma: int) -> list[int]:
    n = len(sequence)
    max_k = 0
    best_i = -1
    for k in range(1, n):
        suffix = sequence[-k:]
        for i in range(n - k):
            if sequence[i:i+k] == suffix:
                if k >= max_k:
                    max_k = k
                    best_i = i

    if max_k == 0:
        return []

    start_idx = best_i + max_k
    return sequence[start_idx : start_idx + gamma]

def get_degenerate_loop_scenario() -> tuple[list[int], int]:
    return [5, 6, 7, 1, 2, 1, 2, 1, 2, 5, 6, 7], 4
