def lock_optimization_profiles(profile_list):
    locked = []
    for p in profile_list:
        locked.append({
            "min_shape": tuple(p["min_shape"]),
            "opt_shape": tuple(p["opt_shape"]),
            "max_shape": tuple(p["max_shape"])
        })
    return locked
