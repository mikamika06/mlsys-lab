import numpy as np


def get_specs():
    specs = [
        {
            "states": ["S0", "S1", "S2"],
            "vocab_size": 5,
            "transitions": [
                ("S0", 0, "S1"),
                ("S0", 1, "S2"),
                ("S1", 2, "S0"),
                ("S1", 3, "S1"),
                ("S2", 4, "S2"),
            ],
        },
        {
            "states": ["A", "B"],
            "vocab_size": 3,
            "transitions": [
                ("A", 0, "B"),
                ("A", 1, "A"),
                ("B", 2, "A"),
            ],
        },
        {
            "states": ["Q0"],
            "vocab_size": 4,
            "transitions": [
                ("Q0", 0, "Q0"),
                ("Q0", 1, "Q0"),
            ],
        },
    ]
    return specs


def build_transition_matrix(spec):
    states = spec["states"]
    vocab_size = spec["vocab_size"]
    transitions = spec["transitions"]
    state_to_idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    mat = np.zeros((n, vocab_size, n), dtype=float)
    for src, token, dst in transitions:
        s_idx = state_to_idx[src]
        d_idx = state_to_idx[dst]
        mat[s_idx, token, d_idx] = 1.0
    for i in range(n):
        row_sum = mat[i].sum(axis=1, keepdims=True)
        row_sum[row_sum == 0.0] = 1.0
        mat[i] = mat[i] / row_sum
    return mat


def compute_acceptance_ceiling(transition_matrix, gamma):
    n, vocab_size, _ = transition_matrix.shape
    start_state = 0
    expected_accepted = 0.0
    current_dist = np.zeros(n)
    current_dist[start_state] = 1.0
    for k in range(gamma):
        next_dist = np.zeros(n)
        step_prob = 0.0
        for s in range(n):
            if current_dist[s] == 0:
                continue
            for token in range(vocab_size):
                token_prob = 1.0 / vocab_size
                dest_probs = transition_matrix[s, token]
                prob_sum = dest_probs.sum()
                if prob_sum > 0:
                    valid_trans = dest_probs / prob_sum
                    for d in range(n):
                        if valid_trans[d] > 0:
                            next_dist[d] += current_dist[s] * token_prob * valid_trans[d]
                            step_prob += current_dist[s] * token_prob * valid_trans[d]
        expected_accepted += step_prob
        current_dist = next_dist
        if current_dist.sum() == 0:
            break
        current_dist /= current_dist.sum()
    return float(expected_accepted)
