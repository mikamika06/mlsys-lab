import sys

sys.path.insert(0, ".")
from eval.compare import check_comparison_validity, is_statistically_significant


def test_detects_invalid_tokenizer_comparison():
    a = {"tokenizer_id": "tok_a", "context_length": 2048, "stride": 512, "dataset_hash": "abc"}
    b = {"tokenizer_id": "tok_b", "context_length": 2048, "stride": 512, "dataset_hash": "abc"}
    res = check_comparison_validity(a, b)
    assert not res["valid"]
    assert "tokenizer_mismatch" in res["reasons"]


def test_hellaswag_error_bars_overlap():
    sig = is_statistically_significant(0.72, 0.02, 0.73, 0.02)
    assert not sig
