import ref
from profiler.merge import merge_profiles, find_straggler, explain_cause

def check(workdir):
    m = {"cause_explained": 0.0}
    profiles = ref.generate_test_data()
    try:
        merged = merge_profiles(profiles)
        s = find_straggler(merged)
        cause = explain_cause(merged, s)
        if cause == "heavy_compute":
            m["cause_explained"] = 1.0
    except Exception:
        pass
    return m
