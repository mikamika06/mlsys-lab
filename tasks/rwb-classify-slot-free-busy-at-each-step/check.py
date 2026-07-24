import numpy as np

def _ref_classify(events, num_steps, num_slots):
    """Independent oracle: replay event log into a full boolean state matrix."""
    state = [-1] * num_slots
    # Group events by step (preserving insertion order within each step)
    by_step = {}
    for step, slot, seq_id in events:
        by_step.setdefault(step, []).append((slot, seq_id))

    result = []
    for t in range(num_steps):
        for slot, seq_id in by_step.get(t, []):
            state[slot] = seq_id
        result.append([state[k] != -1 for k in range(num_slots)])
    return result

def grade(sol, fx) -> dict:
    cases = [
        # (events, num_steps, num_slots)
        ([(0, 0, 1), (2, 0, -1), (1, 1, 2)], 4, 3),
        ([(0, 0, 0), (0, 1, 1)], 3, 2),
        ([], 5, 4),
        ([(0, 2, 5), (3, 2, -1), (3, 0, 7), (5, 0, -1)], 6, 3),
        ([(0, 0, 1), (0, 1, 2), (1, 0, 3), (1, 1, -1),
          (2, 0, -1), (2, 1, 4)], 4, 2),
    ]

    ok = 1.0
    for events, num_steps, num_slots in cases:
        try:
            got = sol.classify_slots(events, num_steps, num_slots)
            got_list = np.asarray(got, dtype=bool).tolist()
        except Exception:
            ok = 0.0
            break
        expected = _ref_classify(events, num_steps, num_slots)
        if got_list != expected:
            ok = 0.0
            break

    return {"exact_match": ok}
