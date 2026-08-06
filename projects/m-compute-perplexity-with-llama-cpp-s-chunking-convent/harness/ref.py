import numpy as np


class SyntheticModel:
    def __init__(self, vocab_size=64, seed=42, noise_std=0.0):
        self.vocab_size = vocab_size
        self.rng = np.random.RandomState(seed)
        self.W = self.rng.randn(vocab_size, vocab_size)
        self.noise_std = noise_std

    def __call__(self, chunk_tokens):
        L = len(chunk_tokens)
        logits = np.zeros((L, self.vocab_size), dtype=np.float32)
        for i, tok in enumerate(chunk_tokens):
            prev = int(tok) % self.vocab_size
            logits[i] = self.W[prev]
            if self.noise_std > 0:
                logits[i] += self.rng.randn(self.vocab_size) * self.noise_std
        return logits


def generate_tokens(num_tokens=128, vocab_size=64, seed=123):
    rng = np.random.RandomState(seed)
    return rng.randint(0, vocab_size, size=num_tokens).tolist()


def ref_compute_perplexity(model, tokens, chunk_size):
    N = len(tokens)
    if N < 2:
        return 0.0
    total_loss = 0.0
    total_targets = 0
    for start in range(0, N, chunk_size):
        chunk = tokens[start : start + chunk_size]
        logits = model(chunk)
        L = len(chunk)
        for j in range(L):
            global_idx = start + j
            target_idx = global_idx + 1
            if target_idx < N:
                t = tokens[target_idx]
                row = np.asarray(logits[j], dtype=np.float64)
                m = np.max(row)
                lse = m + np.log(np.sum(np.exp(row - m)))
                total_loss += lse - row[t]
                total_targets += 1
    if total_targets == 0:
        return 0.0
    return float(np.exp(total_loss / total_targets))


def ref_compute_logit_metrics(base_logits, quant_logits):
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
    top1_acc = float(np.mean(top1_b == top1_q))

    return {"mean_kld": mean_kld, "top1_agreement": top1_acc}
