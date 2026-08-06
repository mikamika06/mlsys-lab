import numpy as np

def compute_attention_entropies(q_seq, k_seq):
    seq_len, head_dim = q_seq.shape
    scale = np.sqrt(head_dim)
    entropies = np.zeros(seq_len)
    for i in range(seq_len):
        logits = np.dot(k_seq[:i+1], q_seq[i]) / scale
        m = np.max(logits)
        p = np.exp(logits - m)
        p = p / np.sum(p)
        p_safe = np.maximum(p, 1e-12)
        entropies[i] = -np.sum(p * np.log(p_safe))
    return entropies
