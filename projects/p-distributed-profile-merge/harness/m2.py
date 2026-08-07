import ref
from profiler.merge import merge_profiles

def check(workdir):
    m = {"profiles_merged": 0.0}
    profiles = ref.generate_test_data()
    try:
        merged = merge_profiles(profiles)
        if "events" in merged and len(merged["events"]) == 4:
            ts_list = [e["ts"] for e in merged["events"]]
            if ts_list == sorted(ts_list):
                m["profiles_merged"] = 1.0
    except Exception:
        pass
    return m
