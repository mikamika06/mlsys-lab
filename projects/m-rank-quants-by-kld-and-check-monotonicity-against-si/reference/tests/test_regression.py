import sys
sys.path.insert(0, ".")
from quval.metrics import rank_quants_by_kld, check_monotonicity, find_disagreements
from quval.stats import derive_sample_size


def test_monotonicity_ideal():
    quants = [
        {"name": "q4", "size_bytes": 400, "logits_ref": [[1.0, 2.0]], "logits_q": [[0.5, 1.5]], "ppl": 10.0},
        {"name": "q8", "size_bytes": 800, "logits_ref": [[1.0, 2.0]], "logits_q": [[0.9, 1.9]], "ppl": 5.0},
    ]
    ranked = rank_quants_by_kld(quants)
    assert check_monotonicity(ranked) is True


def test_sample_size_positive():
    n = derive_sample_size(5.0, 5.5, 1.0)
    assert n > 0


def test_disagreements_list():
    quants = [
        {"name": "a", "size_bytes": 400, "logits_ref": [[1.0, 2.0]], "logits_q": [[0.5, 1.5]], "ppl": 10.0},
        {"name": "b", "size_bytes": 800, "logits_ref": [[1.0, 2.0]], "logits_q": [[0.9, 1.9]], "ppl": 5.0},
    ]
    res = find_disagreements(quants)
    assert isinstance(res, list)
