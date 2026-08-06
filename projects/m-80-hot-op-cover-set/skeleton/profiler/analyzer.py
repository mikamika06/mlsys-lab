def detect_warmup(events):
    """
    Returns the timestamp (ts) of the first 'model_run' event whose duration
    is <= 1.2 * median duration of all 'model_run' events.
    If no 'model_run' events exist, return 0.
    """
    raise NotImplementedError


def hot_op_cover(events, threshold=0.8):
    """
    Returns a set of op_names that account for at least `threshold` of total
    node duration in the steady-state (post-warmup) trace.
    Ties in duration should be resolved by sorting op_names alphabetically.
    """
    raise NotImplementedError


def attribute_speedup(events_before, events_after):
    """
    Returns a dictionary mapping each op_name to the average time saved per run:
    (average duration per model_run in 'before') - (average duration in 'after').
    Only consider steady-state events.
    """
    raise NotImplementedError
