import numpy as np

def get_transition_probs(logits):
    exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_l / np.sum(exp_l, axis=1, keepdims=True)

def sample_teacher(logits, start_state, steps, seed=42):
    rng = np.random.RandomState(seed)
    probs = get_transition_probs(logits)
    V = logits.shape[0]
    seq = [start_state]
    curr = start_state
    for _ in range(steps - 1):
        curr = rng.choice(V, p=probs[curr])
        seq.append(curr)
    return np.array(seq)
