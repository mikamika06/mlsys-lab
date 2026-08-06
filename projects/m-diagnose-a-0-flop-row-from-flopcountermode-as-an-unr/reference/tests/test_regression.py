from flopdiag.mfu import rank_mfu


def test_mfu_ranking():
    records = [
        {"id": "a", "tokens_per_sec": 1000, "params": 7e9, "peak_tflops": 312},
        {"id": "b", "tokens_per_sec": 2000, "params": 7e9, "peak_tflops": 312},
    ]
    ranked = rank_mfu(records)
    assert ranked == ["b", "a"]
