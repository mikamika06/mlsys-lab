import sys

sys.path.insert(0, ".")
from rundiag import repetition_report, num_predict_budget

TOKENS = [5, 1, 2, 1, 3, 1, 4, 1, 1, 6]


def test_report_positions_match_token_occurrences():
    rep = repetition_report(TOKENS, window=5, threshold=3)
    assert rep["triggered"] is True
    assert rep["positions"] == [i for i, t in enumerate(TOKENS) if t == rep["token"]]


def test_histogram_sums_to_total_tokens():
    rep = repetition_report(TOKENS, window=5, threshold=3)
    assert sum(rep["histogram"].values()) == rep["total_tokens"] == len(TOKENS)


def test_explicit_predict_never_exceeds_request():
    got = num_predict_budget(10, prompt_tokens=100, context_size=1000, hard_cap=1000)
    assert got <= 10


def test_infinite_generation_not_bounded_by_context():
    fill = num_predict_budget(-2, prompt_tokens=100, context_size=4096, hard_cap=1_000_000)
    infinite = num_predict_budget(-1, prompt_tokens=100, context_size=4096, hard_cap=1_000_000)
    assert infinite > fill, f"num_predict=-1 gave {infinite}, expected more than fill-context {fill}"
