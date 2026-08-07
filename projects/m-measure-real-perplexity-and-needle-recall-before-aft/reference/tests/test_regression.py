import sys
import numpy as np

sys.path.insert(0, ".")
from longctx.scaling import apply_scaling
from longctx.perplexity import measure_perplexity
from longctx.needle import measure_recall_at_k

def test_scaling_monotone_reduction():
    pos = np.array([100, 200, 300])
    s1 = apply_scaling(pos, 1.0)
    s2 = apply_scaling(pos, 2.0)
    assert np.all(s2 < s1)

def test_perplexity_positive():
    logits = np.random.randn(10, 32)
    targets = np.random.randint(0, 32, size=(10,))
    ppl = measure_perplexity(logits, targets, 1.0)
    assert ppl > 0.0

def test_recall_bounds():
    scores = np.random.randn(5, 50)
    needles = np.array([10, 20, 30, 40, 5])
    recall = measure_recall_at_k(scores, needles, 5)
    assert 0.0 <= recall <= 1.0
