import tempfile
import ref


def check(workdir):
    from leak.detector import MemorySnapshotAnalyzer

    m = {"snapshot_loaded": 0.0}
    with tempfile.TemporaryDirectory() as tmp:
        path = ref.generate_snapshot_file(tmp)
        analyzer = MemorySnapshotAnalyzer()
        try:
            res = analyzer.load_snapshot(path)
            if res and analyzer.data.get("active") == 100:
                m["snapshot_loaded"] = 1.0
        except Exception:
            pass
    return m
