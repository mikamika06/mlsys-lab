import numpy as np
from .chunking import compute_perplexity
from .metrics import compute_logit_metrics


def dump_f16_logits(model, tokens, chunk_size, output_path):
    """Run model on token chunks and dump concatenated F16 logits to file."""
    N = len(tokens)
    all_logits = []

    for start in range(0, N, chunk_size):
        chunk = tokens[start : start + chunk_size]
        logits = model(chunk)
        all_logits.append(np.asarray(logits, dtype=np.float32))

    if all_logits:
        concatenated = np.concatenate(all_logits, axis=0)
    else:
        concatenated = np.empty((0, 0), dtype=np.float32)

    f16_logits = concatenated.astype(np.float16)
    np.save(output_path, f16_logits)
    return output_path


def score_quantized_model(quant_model, tokens, chunk_size, f16_logits_path):
    """Score quantized model against reference F16 logits."""
    f16_logits = np.load(f16_logits_path).astype(np.float32)

    N = len(tokens)
    quant_logits_list = []

    for start in range(0, N, chunk_size):
        chunk = tokens[start : start + chunk_size]
        logits = quant_model(chunk)
        quant_logits_list.append(np.asarray(logits, dtype=np.float32))

    if quant_logits_list:
        quant_logits = np.concatenate(quant_logits_list, axis=0)
    else:
        quant_logits = np.empty((0, 0), dtype=np.float32)

    ppl = compute_perplexity(quant_model, tokens, chunk_size)
    metrics = compute_logit_metrics(f16_logits, quant_logits)

    return {
        "perplexity": float(ppl),
        "mean_kld": float(metrics["mean_kld"]),
        "top1_agreement": float(metrics["top1_agreement"]),
    }
