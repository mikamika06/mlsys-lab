import numpy as np

def generate_fixtures():
    np.random.seed(42)
    logits_jsd = np.random.randn(10, 50).astype(np.float32)
    logits_fkl = np.random.randn(10, 50).astype(np.float32)
    tokens_jsd = np.random.randint(0, 50, size=(10, 20))
    tokens_fkl = np.random.randint(0, 50, size=(10, 20))
    return logits_jsd, logits_fkl, tokens_jsd, tokens_fkl

def compute_entropy(logits):
    exps = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exps / np.sum(exps, axis=-1, keepdims=True)
    ent = -np.sum(probs * np.log(probs + 1e-12), axis=-1)
    return float(np.mean(ent))

def compute_diversity_ratio(logits_jsd, logits_fkl, tokens_jsd, tokens_fkl):
    ent_jsd = compute_entropy(logits_jsd)
    ent_fkl = compute_entropy(logits_fkl)
    uniq_jsd = len(np.unique(tokens_jsd)) / tokens_jsd.size
    uniq_fkl = len(np.unique(tokens_fkl)) / tokens_fkl.size
    div_jsd = ent_jsd * uniq_jsd
    div_fkl = ent_fkl * uniq_fkl
    ratio = div_jsd / (div_fkl + 1e-12)
    return float(ratio)
