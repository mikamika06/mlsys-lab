import numpy as np


def build_transition_matrix(grammar_spec):
    states = grammar_spec["states"]
    vocab_size = grammar_spec["vocab_size"]
    transitions = grammar_spec["transitions"]
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
