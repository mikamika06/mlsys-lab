import ref


def check(workdir):
    from leak.detector import MemorySnapshotAnalyzer

    m = {"stable_run": 0.0}
    analyzer = MemorySnapshotAnalyzer()
    try:
        growth_before = analyzer.simulate_epoch()
        analyzer.fix_retention()
        growth_after = analyzer.simulate_epoch()
        if growth_before > 0 and growth_after == 0:
            m["stable_run"] = 1.0
    except Exception:
        pass
    return m
