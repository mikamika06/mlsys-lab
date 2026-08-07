import ref


def check(workdir):
    from leak.detector import MemorySnapshotAnalyzer

    m = {"retention_fixed": 0.0}
    analyzer = MemorySnapshotAnalyzer()
    try:
        res = analyzer.fix_retention()
        if res and analyzer.fixed:
            m["retention_fixed"] = 1.0
    except Exception:
        pass
    return m
