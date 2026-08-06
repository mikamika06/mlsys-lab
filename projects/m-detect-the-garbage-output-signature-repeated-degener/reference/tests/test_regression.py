from engine.detector import detect_garbage
from engine.blast import estimate_blast


def test_detector_basic():
    tokens = ["hello", "world", "!", "!", "!", "!", "!"]
    assert detect_garbage(tokens) is True


def test_detector_normal():
    tokens = ["the", "quick", "brown", "fox", "jumps"]
    assert detect_garbage(tokens) is False


def test_blast_basic():
    reqs = [
        {"id": "r1", "index": 0, "max_retries": 3, "retry_count": 0},
        {"id": "r2", "index": 1, "max_retries": 3, "retry_count": 1}
    ]
    res = estimate_blast(reqs, 1)
    assert res["lost_count"] == 1
    assert res["retried_count"] == 1
