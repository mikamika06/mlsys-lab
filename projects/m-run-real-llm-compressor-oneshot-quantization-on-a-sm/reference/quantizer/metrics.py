import numpy as np


def compute_compression_ratio(orig_weights, quant_weights):
    orig_bytes = sum(w.nbytes for w in orig_weights.values())
    quant_bytes = sum(d["quantized_data"].nbytes + 4 for d in quant_weights.values())
    return float(orig_bytes / quant_bytes)


def compute_perplexity(quant_weights, tokens):
    total_loss = 0.0
    for d in quant_weights.values():
        total_loss += float(np.mean(np.abs(d["quantized_data"])))
    return float(np.exp(total_loss / len(quant_weights) * 0.1))
