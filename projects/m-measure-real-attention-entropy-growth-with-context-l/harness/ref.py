import numpy as np

def get_m1_data():
    rng = np.random.RandomState(42)
    q = rng.randn(64, 32).astype(np.float32)
    k = rng.randn(64, 32).astype(np.float32)
    return q, k

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

def get_m2_data():
    accs = np.array([
        [0.99, 0.95, 0.90],
        [0.98, 0.80, 0.10],
        [0.95, 0.50, 0.05]
    ])
    ents = np.array([
        [2.0, 3.0, 4.0],
        [2.1, 8.5, 12.0],
        [2.0, 2.5, 3.0]
    ])
    lengths = np.array([4000, 16000, 64000])
    return accs, ents, lengths, 0.5, 10.0

def diagnose_models(accuracies, max_entropies, lengths, acc_thresh, ent_thresh):
    out = []
    mean_x = np.mean(lengths)
    var_x = np.sum((lengths - mean_x)**2)
    num_models = accuracies.shape[0]

    for i in range(num_models):
        y = accuracies[i]
        mean_y = np.mean(y)
        slope = np.sum((lengths - mean_x) * (y - mean_y)) / var_x
        final_acc = accuracies[i, -1]
        final_ent = max_entropies[i, -1]
        if final_acc >= acc_thresh:
            mode = 'none'
        elif final_ent > ent_thresh:
            mode = 'dilution'
        else:
            mode = 'rope'
        out.append({'model_idx': i, 'slope': float(slope), 'mode': mode})
    return out
