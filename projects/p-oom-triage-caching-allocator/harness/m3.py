import ref

def check(workdir):
    m = {"leak_found": 0.0}
    try:
        from oom_triage.analyzer import find_leaked_tensors
        snaps = ref.generate_test_snapshots()
        leaks = find_leaked_tensors(snaps)
        if leaks == [2]:
            m["leak_found"] = 1.0
    except Exception:
        pass
    return m
