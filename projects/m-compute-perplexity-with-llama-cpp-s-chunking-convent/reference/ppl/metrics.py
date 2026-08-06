import numpy as np


def compute_logit_metrics(base_logits, quant_logits):
    """Compute mean KL divergence and top-1 agreement between base and quant logits."""
    base_logits = np.asarray(base_logits, dtype=np.float64)
    quant_logits = np.asarray(quant_logits, dtype=np.float64)

    N = base_logits.shape[0]
    if N == 0:
        return {"mean_kld": 0.0, "top1_agreement": 0.0}

    mb = np.max(base_logits, axis=-1, keepdims=True)
    lse_b = mb + np.log(np.sum(np.exp(base_logits - mb), axis=-1, keepdims=True))
    log_p = base_logits - lse_b
    p = np.exp(log_p)

    mq = np.max(quant_logits, axis=-1, keepdims=True)
    lse_q = mq + np.log(np.sum(np.exp(quant_logits - mq), axis=-1, keepdims=True))
    log_q = quant_logits - lse_q

    kld_per_pos = np.sum(p * (log_p - log_q), axis=-1)
    mean_kld = float(np.mean(kld_per_pos))

    top1_b = np.argmax(base_logits, axis=-1)
    top1_q = np.argmax(quant_logits, axis=-1)
    top1_agreement = float(np.mean(top1_b == top1_q))

    return {"mean_kld": mean_kld, "top1_agreement": top1_agreement}
