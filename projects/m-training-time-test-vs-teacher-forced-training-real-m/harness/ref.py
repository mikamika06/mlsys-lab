import numpy as np

np.random.seed(42)

CONFIGS = []
for i in range(3):
    seq_len = 16
    vocab_size = 32
    draft_logits = np.random.randn(seq_len, vocab_size).astype(np.float32)
    target_logits = np.random.randn(seq_len, vocab_size).astype(np.float32)
    tokens = np.random.randint(0, vocab_size, size=(seq_len,)).tolist()
    CONFIGS.append({
        "tokens": tokens,
        "draft_logits": draft_logits,
        "target_logits": target_logits,
        "gamma": 4
    })


def compute_teacher_forced_loss(tokens, draft_logits):
    import numpy as np
    logits = draft_logits[:-1]
    targets = tokens[1:]
    exps = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exps / np.sum(exps, axis=-1, keepdims=True)
    nll = -np.log(probs[np.arange(len(targets)), targets] + 1e-12)
    return float(np.mean(nll))


def simulate_acceptance_rate(tokens, draft_logits, target_logits, gamma=4):
    import numpy as np
    accepted_total = 0
    total_proposed = 0
    n = len(tokens)
    idx = 0
    while idx < n - gamma:
        k = min(gamma, n - idx)
        for j in range(k):
            total_proposed += 1
            p_draft = np.exp(draft_logits[idx + j] - np.max(draft_logits[idx + j]))
            p_draft /= np.sum(p_draft)
            p_target = np.exp(target_logits[idx + j] - np.max(target_logits[idx + j]))
            p_target /= np.sum(p_target)
            token = tokens[idx + j]
            ratio = p_target[token] / (p_draft[token] + 1e-12)
            if ratio >= 1.0 or ratio >= 0.5:
                accepted_total += 1
            else:
                break
        idx += max(1, k)
    if total_proposed == 0:
        return 0.0
    return float(accepted_total / total_proposed)
