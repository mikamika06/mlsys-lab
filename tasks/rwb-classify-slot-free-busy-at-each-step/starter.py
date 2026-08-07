def classify_slots(events, num_steps, num_slots):
    """Classify each (step, slot) as busy or free from sparse assignment events.

    Args:
        events: list of (step, slot, seq_id) tuples.
                seq_id >= 0 means assigned; seq_id == -1 means freed.
        num_steps: total number of time steps (0 .. num_steps-1).
        num_slots: number of slots (0 .. num_slots-1).

    Returns:
        list[float] of shape (num_steps, num_slots), dtype bool.
        True = busy, False = free.
    """
    raise NotImplementedError('your code here')
