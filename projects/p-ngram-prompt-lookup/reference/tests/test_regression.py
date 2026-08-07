import sys

sys.path.insert(0, ".")
from ngram.index import PromptNgramIndex
from ngram.policy import select_candidates, should_disable
from ngram.engine import NgramSpeculativeEngine

def test_index_basic():
    prompt = [1, 2, 3, 4, 1, 2, 3, 5]
    idx = PromptNgramIndex(prompt, n=3)
    res = idx.lookup([1, 2, 3])
    assert len(res) > 0

def test_policy_selection():
    prompt = [10, 20, 30, 40, 50, 10, 20, 30, 60]
    idx = PromptNgramIndex(prompt, n=3)
    cands = select_candidates(idx, [10, 20, 30], k=3)
    assert len(cands) == 3

def test_disable_logic():
    history = [0.0] * 25
    assert should_disable(history, threshold=0.1) is True

def test_engine_regression():
    prompt = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
    engine = NgramSpeculativeEngine(prompt, k=2, disable_threshold=0.05)
    mock_model = lambda out, spec=None: ([spec[0]] if spec else [99], 1 if spec else 0)
    out = engine.run(mock_model, max_steps=5)
    assert len(out) > 0
