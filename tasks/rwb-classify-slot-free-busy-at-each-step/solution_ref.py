def classify_slots(events, num_steps, num_slots):
    """Reconstruct and classify slot states from sparse assignment events.

    Returns a list of lists of booleans of shape (num_steps, num_slots) where True means
    the slot is busy and False means free.
    """
    state = [-1] * num_slots

    # Group events by step
    by_step = {}
    for step, slot, seq_id in events:
        by_step.setdefault(step, []).append((slot, seq_id))

    result = []

    for t in range(num_steps):
        for slot, seq_id in by_step.get(t, []):
            state[slot] = seq_id
        result.append([state[k] != -1 for k in range(num_slots)])

    return result
