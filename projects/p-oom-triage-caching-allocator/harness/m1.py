import os
import ref

def check(workdir):
    m = {"snapshot_loaded": 0.0}
    path = os.path.join(workdir, "test_snap.json")
    ref.create_mock_snapshot(path)
    try:
        from oom_triage.analyzer import load_snapshot
        data = load_snapshot(path)
        if isinstance(data, dict) and "segments" in data:
            m["snapshot_loaded"] = 1.0
    except Exception:
        pass
    finally:
        if os.path.exists(path):
            os.remove(path)
    return m
