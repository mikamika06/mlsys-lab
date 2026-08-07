import ref

def check(workdir):
    m = {"leak_detected": 0.0}
    try:
        from qlora_fix.memory import detect_growth
        hist_growing = [1000, 1050, 1100, 1200]
        if detect_growth(hist_growing) == ref.oracle_detect_growth(hist_growing):
            m["leak_detected"] = 1.0
    except Exception:
        pass
    return m
