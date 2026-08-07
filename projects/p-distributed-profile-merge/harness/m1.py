import ref
from profiler.merge import align_clocks

def check(workdir):
    m = {"clock_sync_ok": 0.0}
    profiles = ref.generate_test_data()
    try:
        aligned = align_clocks(profiles)
        if len(aligned) == 2:
            syncs = [[e["ts"] for e in p["events"] if e["name"] == "sync"] for p in aligned]
            if syncs[0][0] == syncs[1][0]:
                m["clock_sync_ok"] = 1.0
    except Exception:
        pass
    return m
