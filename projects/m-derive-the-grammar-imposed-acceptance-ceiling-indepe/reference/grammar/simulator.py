import numpy as np


def simulate_acceptance(matrix, initial_state, draft_sequences):
    n_states, vocab_size, _ = matrix.shape
    results = []
    for seq in draft_sequences:
        state = initial_state
        accepted = 0
        for token in seq:
            if not (0 <= token < vocab_size):
                break
            next_states = matrix[state, token]
            valid_next = np.where(next_states > 0)[0]
            if len(valid_next) == 0:
                break
            state = int(valid_next[0])
            accepted += 1
        results.append(accepted / max(1, len(seq)))
    return float(np.mean(results))
