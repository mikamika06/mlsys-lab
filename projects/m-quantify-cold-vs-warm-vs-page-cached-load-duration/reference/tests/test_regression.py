from keepalive.policy import select_keep_alive

def test_select_keep_alive_respects_memory_cap():
    models = [
        {"id": "m1", "size_mb": 500},
        {"id": "m2", "size_mb": 1000},
        {"id": "m3", "size_mb": 300}
    ]
    freqs = {"m1": 10.0, "m2": 5.0, "m3": 20.0}
    selected = select_keep_alive(models, 1200, freqs)
    total_mem = sum(next(m["size_mb"] for m in models if m["id"] == sid) for sid in selected)
    assert total_mem <= 1200

def test_select_keep_alive_prioritizes_high_value():
    models = [
        {"id": "m1", "size_mb": 500},
        {"id": "m2", "size_mb": 500}
    ]
    freqs = {"m1": 1.0, "m2": 10.0}
    selected = select_keep_alive(models, 500, freqs)
    assert selected == ["m2"]
