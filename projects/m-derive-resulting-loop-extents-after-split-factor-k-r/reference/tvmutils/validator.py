class ScheduleError(Exception):
    pass


def verify_schedule_behavior(extent, factor):
    outer = (extent + factor - 1) // factor
    inner = factor
    return {"outer": outer, "inner": inner, "product": outer * inner}


def trigger_vectorization_error(extent):
    if extent > 1:
        raise ScheduleError("ScheduleError: Vectorize failed due to data dependency on loop axis")
    return True
