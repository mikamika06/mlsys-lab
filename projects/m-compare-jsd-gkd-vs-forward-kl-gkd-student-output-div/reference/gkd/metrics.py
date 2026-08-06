import numpy as np
from gkd.diversity import compute_token_entropy, compute_vocabulary_coverage

def compare_gkd_diversity(logits_jsd, logits_fkl, tokens_jsd, tokens_fkl, vocab_size=50):
    ent_jsd = compute_token_entropy(logits_jsd)
    ent_fkl = compute_token_entropy(logits_fkl)
    cov_jsd = compute_vocabulary_coverage(tokens_jsd, vocab_size)
    cov_fkl = compute_vocabulary_coverage(tokens_fkl, vocab_size)
    div_jsd = ent_jsd * cov_jsd
    div_fkl = ent_fkl * cov_fkl
    ratio = div_jsd / (div_fkl + 1e-12)
    return {
        "div_jsd": float(div_jsd),
        "div_fkl": float(div_fkl),
        "ratio": float(ratio)
    }
