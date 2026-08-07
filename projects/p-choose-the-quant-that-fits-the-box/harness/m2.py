def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from quant.analyzer import measure_peak_memory
    import ref

    m = {"peak_memory_measured": 0.0}
    try:
        val = measure_peak_memory(7000000000, 4.0, 2048.0)
        oval = ref.oracle_measure_peak_memory(7000000000, 4.0, 2048.0)
        if abs(val - oval) < 1e-5:
            m["peak_memory_measured"] = 1.0
    except Exception:
        pass
    return m
