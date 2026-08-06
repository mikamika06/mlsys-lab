from sharder.schedule import compute_conversion_schedule

def test_sharding_constraints():
    vocab = {"tokens": ["a", "b"], "scores": [1.0, 2.0]}
    tensors = [
        {"name": "t1", "shape": (100, 100), "dtype": "float32"},
        {"name": "t2", "shape": (100, 100), "dtype": "float32"},
    ]
    max_bytes = 40000
    res = compute_conversion_schedule(vocab, tensors, max_bytes)
    assert len(res["shards"]) > 1
