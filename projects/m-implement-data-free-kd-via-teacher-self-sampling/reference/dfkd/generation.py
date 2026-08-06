import numpy as np

def sample_teacher(teacher_probs, start, steps, seed=42):
    rng = np.random.RandomState(seed)
    V = teacher_probs.shape[0]
    seq = [start]
    curr = start
    for _ in range(steps - 1):
        curr = rng.choice(V, p=teacher_probs[curr])
        seq.append(curr)
    return np.array(seq)

def measure_diversity(sequence):
    return len(set(sequence))
