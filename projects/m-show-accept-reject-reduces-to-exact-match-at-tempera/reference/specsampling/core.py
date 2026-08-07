import numpy as np

def softmax(x, T):
    if T == 0.0:
        res = np.zeros_like(x)
        res[np.argmax(x)] = 1.0
        return res
    ex = np.exp((x - np.max(x)) / T)
    return ex / np.sum(ex)

def verify_zero_temp_reduction(target_logits, draft_logits):
    p = softmax(target_logits, 0.0)
    q = softmax(draft_logits, 0.0)
    target_argmax = int(np.argmax(target_logits))
    draft_argmax = int(np.argmax(draft_logits))
    exact_match = (target_argmax == draft_argmax)
    accept_prob = min(1.0, p[draft_argmax] / q[draft_argmax]) if q[draft_argmax] > 0 else 0.0
    return bool(exact_match and (accept_prob == 1.0 if exact_match else accept_prob == 0.0))

def measure_acceptance_rates(target_logits, draft_logits, temps):
    rates = []
    np.random.seed(42)
    for T in temps:
        p = softmax(target_logits, T)
        q = softmax(draft_logits, T)
        accepts = []
        for _ in range(1000):
            token = np.random.choice(len(q), p=q)
            ratio = p[token] / q[token] if q[token] > 0 else 0.0
            acc = min(1.0, ratio)
            accepts.append(np.random.rand() < acc)
        rates.append(float(np.mean(accepts)))
    return rates

def quantify_mismatch_skew(target_logits, draft_logits, draft_T, target_T):
    p = softmax(target_logits, target_T)
    q_wrong = softmax(draft_logits, draft_T)
    kl = np.sum(p * np.log((p + 1e-10) / (q_wrong + 1e-10)))
    return float(kl)
