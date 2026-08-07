import ref

def check(workdir):
    m = {"profiles_fixed": 0.0}
    try:
        from trt_builder.profiles import lock_optimization_profiles
        profiles = ref.get_sample_profiles()
        locked = lock_optimization_profiles(profiles)
        if isinstance(locked, list) and len(locked) == 1:
            m["profiles_fixed"] = 1.0
    except Exception:
        pass
    return m
