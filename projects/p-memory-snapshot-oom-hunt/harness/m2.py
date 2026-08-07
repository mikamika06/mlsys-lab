import ref


def check(workdir):
    from leak.detector import MemorySnapshotAnalyzer

    m = {"fragmentation_distinguished": 0.0}
    analyzer = MemorySnapshotAnalyzer()
    base = {"active": 100, "allocated": 150}
    curr_leak = {"active": 140, "allocated": 160}
    curr_frag = {"active": 100, "allocated": 200}
    try:
        res_leak = analyzer.analyze_fragmentation(base, curr_leak)
        res_frag = analyzer.analyze_fragmentation(base, curr_frag)
        if not res_leak["fragmented"] and res_frag["fragmented"]:
            m["fragmentation_distinguished"] = 1.0
    except Exception:
        pass
    return m
