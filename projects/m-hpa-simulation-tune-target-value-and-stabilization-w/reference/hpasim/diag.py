def diagnose_thrash(timeline):
    scale_downs = 0
    rapid_downs = 0
    prev_time = None
    prev_reps = None
    for t, reps in timeline:
        if prev_reps is not None and reps < prev_reps:
            scale_downs += 1
            if (t - prev_time) < 60:
                rapid_downs += 1
        prev_time = t
        prev_reps = reps

    if rapid_downs > 0:
        return "stabilization_window"
    if scale_downs > 2:
        return "target_value"
    return "none"
