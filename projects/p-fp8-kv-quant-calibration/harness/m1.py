def check(workdir):
    from kvquant.cache import KVCacheTracker
    m = {"baseline_tracked": 0.0, "fp16_bytes_correct": 0.0}

    tracker = KVCacheTracker(32, 8, 128)
    b = tracker.record(2, 100)
    expected = 32 * 2 * 8 * 128 * 2 * 2 * 100

    if b == expected:
        m["baseline_tracked"] = 1.0
        m["fp16_bytes_correct"] = 1.0
    return m
