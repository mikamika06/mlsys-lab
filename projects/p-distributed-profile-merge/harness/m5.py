import ref
from profiler.merge import merge_profiles, find_straggler, confirm_straggler

def check(workdir):
    m = {"straggler_confirmed": 0.0}
    profiles = ref.generate_test_data()
    try:
        merged = merge_profiles(profiles)
        s = find_straggler(merged)
        if confirm_straggler(merged, s):
            m["straggler_confirmed"] = 1.0
    except Exception:
        pass
    return m
