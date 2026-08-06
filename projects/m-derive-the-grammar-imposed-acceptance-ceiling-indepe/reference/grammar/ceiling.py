import numpy as np


def compute_acceptance_ceiling(transition_matrix, gamma):
    n, vocab_size, _ = transition_matrix.shape
    start_state = 0
    p_accept = 1.0
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
