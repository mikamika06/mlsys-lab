import sys
sys.path.insert(0, ".")
from benchopt.validation import check_tg_ordering

def test_tg_ordering_matches_bytes_read():
    runs = [
        {"bytes_read": 1000, "tg_throughput": 10.0},
        {"bytes_read": 2000, "tg_throughput": 15.0},
        {"bytes_read": 3000, "tg_throughput": 20.0},
    ]
    assert check_tg_ordering(runs) is True

def test_tg_ordering_fails_on_inverted():
    runs = [
        {"bytes_read": 1000, "tg_throughput": 20.0},
        {"bytes_read": 2000, "tg_throughput": 10.0},
    ]
    assert check_tg_ordering(runs) is False
