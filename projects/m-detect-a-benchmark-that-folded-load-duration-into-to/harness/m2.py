import ref

def check(workdir):
    from runner.audit import detect_folded_benchmarks
    bms = ref.get_test_benchmarks()
    try:
        detected = detect_folded_benchmarks(bms)
        expected = ["bm-folded"]
        match = (set(detected) == set(expected))
    except Exception:
        match = False
    return {"detection_matched": 1.0 if match else 0.0}
