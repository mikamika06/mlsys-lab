def fused_cross_entropy(logits, targets):
    import numpy as np
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    max_logits = np.max(logits, axis=1, keepdims=True)
    stable = logits - max_logits
    exp_sum = np.sum(np.exp(stable), axis=1, keepdims=True)
    logsumexp = np.log(exp_sum) + max_logits.squeeze()
    batch_idx = np.arange(logits.shape[0])
    target_logit = logits[batch_idx, targets]
    ce = - (target_logit - logsumexp)
    return float(np.mean(ce))
