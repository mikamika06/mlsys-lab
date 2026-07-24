def gpipe_bubble_fraction(microbatches: int, stages: int) -> float:
    """Fraction of GPipe's per-device time slots spent idle (the "bubble").

    See task.md for the derivation. Formula: (p - 1) / (m + p - 1).
    """
    m = microbatches
    p = stages
    idle_slots = p - 1
    total_slots = m + p - 1
    return idle_slots / total_slots
