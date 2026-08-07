import ref
from profiler.merge import merge_profiles, find_straggler

def check(workdir):
    m = {"straggler_identified": 0.0}
    profiles = ref.generate_test_data()
    try:
        merged = merge_profiles(profiles)
        straggler = find_straggler(merged)
        if straggler == 1:
            m["straggler_identified"] = 1.0
    except Exception:
        pass
    return m
