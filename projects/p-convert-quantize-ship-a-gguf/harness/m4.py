import sys
import numpy as np
import ref

def check(workdir):
    out = {"fp16_kld_zero": 0.0, "q8_kld_lower_than_q4": 0.0, "ppl_ordered": 0.0}

    sys.path.insert(0, workdir)
    from gguf_pipeline.evaluator import compute_perplexity, compute_kl_divergence

    np.random.seed(42)
    fp16_logits = np.random.randn(20, 100)
    q8_logits = fp16_logits + np.random.normal(0, 0.05, size=fp16_logits.shape)
    q4_logits = fp16_logits + np.random.normal(0, 0.2, size=fp16_logits.shape)

    targets = np.random.randint(0, 100, size=20)

    kld_self = compute_kl_divergence(fp16_logits, fp16_logits)
    if abs(kld_self) < 1e-5:
        out["fp16_kld_zero"] = 1.0

    kld_q8 = compute_kl_divergence(fp16_logits, q8_logits)
    kld_q4 = compute_kl_divergence(fp16_logits, q4_logits)
    if kld_q8 < kld_q4:
        out["q8_kld_lower_than_q4"] = 1.0

    ppl_fp16 = compute_perplexity(fp16_logits, targets)
    ppl_q4 = compute_perplexity(q4_logits, targets)
    if ppl_fp16 <= ppl_q4:
        out["ppl_ordered"] = 1.0

    return out
