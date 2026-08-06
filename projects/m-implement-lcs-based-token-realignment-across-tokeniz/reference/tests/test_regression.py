import sys
sys.path.insert(0, ".")
from realignment.align import align_tokens
from realignment.metrics import compute_metrics
from realignment.evaluate import evaluate_uad

def test_lcs_identical_sequences():
    tokens = [10, 20, 30, 40]
    mapping = align_tokens(tokens, tokens)
    assert len(mapping) == len(tokens)
    assert mapping == [(0, 0), (1, 1), (2, 2), (3, 3)]

def test_acceptance_rate_bounds():
    draft = [1, 2, 3]
    target = [1, 9, 3]
    mapping = align_tokens(draft, target)
    metrics = compute_metrics(draft, target, mapping, 1.0)
    assert 0.0 <= metrics["acceptance_rate"] <= 1.0

def test_evaluate_uad_keys():
    cfg = {"draft_tokens": [1, 2], "target_tokens": [1, 2], "family": "same"}
    res = evaluate_uad(cfg)
    assert "mapping" in res
    assert "metrics" in res
    assert "worth_it" in res
