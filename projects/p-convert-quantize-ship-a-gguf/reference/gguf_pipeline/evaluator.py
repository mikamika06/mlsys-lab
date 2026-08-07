import numpy as np

def compute_perplexity(logits, target_ids):
    logits = np.asarray(logits, dtype=np.float64)
    target_ids = np.asarray(target_ids, dtype=np.int64)
    shift_logits = logits[:-1] if logits.shape[0] == target_ids.shape[0] else logits
    shift_targets = target_ids[1:] if logits.shape[0] == target_ids.shape[0] else target_ids

    exp_logits = np.exp(shift_logits - np.max(shift_logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    nll = []
    for i, target in enumerate(shift_targets):
        p = np.clip(probs[i, target], 1e-12, 1.0)
        nll.append(-np.log(p))
    return float(np.exp(np.mean(nll)))

def compute_kl_divergence(p_logits, q_logits):
    p_logits = np.asarray(p_logits, dtype=np.float64)
    q_logits = np.asarray(q_logits, dtype=np.float64)

    p_exp = np.exp(p_logits - np.max(p_logits, axis=-1, keepdims=True))
    p_probs = p_exp / np.sum(p_exp, axis=-1, keepdims=True)

    q_exp = np.exp(q_logits - np.max(q_logits, axis=-1, keepdims=True))
    q_probs = q_exp / np.sum(q_exp, axis=-1, keepdims=True)

    p_probs = np.clip(p_probs, 1e-12, 1.0)
    q_probs = np.clip(q_probs, 1e-12, 1.0)

    kl = np.sum(p_probs * (np.log(p_probs) - np.log(q_probs)), axis=-1)
    return float(np.mean(kl))

def evaluate_model_quality(model_fn, dataset):
    ppls = []
    for sample in dataset:
        logits = model_fn(sample["input_ids"])
        ppl = compute_perplexity(logits, sample["target_ids"])
        ppls.append(ppl)
    return float(np.mean(ppls))
