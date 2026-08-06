import numpy as np
from gkd.diversity import compute_token_entropy, compute_vocabulary_coverage
from gkd.metrics import compare_gkd_diversity

def test_entropy_properties():
    logits = np.zeros((2, 10), dtype=np.float32)
    ent = compute_token_entropy(logits)
    assert np.isclose(ent, np.log(10), atol=1e-5)

def test_diversity_ratio_bounds():
    logits_jsd = np.ones((5, 20), dtype=np.float32)
    logits_fkl = np.ones((5, 20), dtype=np.float32)
    tokens_jsd = np.arange(100).reshape(5, 20)
    tokens_fkl = np.arange(100).reshape(5, 20)
    res = compare_gkd_diversity(logits_jsd, logits_fkl, tokens_jsd, tokens_fkl, vocab_size=100)
    assert res["ratio"] > 0.0
