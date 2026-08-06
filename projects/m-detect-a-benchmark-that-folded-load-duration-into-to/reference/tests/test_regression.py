from runner.audit import is_load_folded, detect_folded_benchmarks

def test_is_load_folded_positive():
    b = {
        "id": "test-1",
        "total_tokens": 100,
        "generation_duration": 1.0,
        "load_duration": 1.0,
        "reported_tok_s": 50.0
    }
    assert is_load_folded(b) is True

def test_is_load_folded_negative():
    b = {
        "id": "test-2",
        "total_tokens": 100,
        "generation_duration": 1.0,
        "load_duration": 1.0,
        "reported_tok_s": 100.0
    }
    assert is_load_folded(b) is False

def test_detect_folded_benchmarks():
    benchmarks = [
        {
            "id": "b1",
            "total_tokens": 200,
            "generation_duration": 2.0,
            "load_duration": 2.0,
            "reported_tok_s": 50.0
        },
        {
            "id": "b2",
            "total_tokens": 200,
            "generation_duration": 2.0,
            "load_duration": 2.0,
            "reported_tok_s": 100.0
        }
    ]
    assert detect_folded_benchmarks(benchmarks) == ["b1"]
