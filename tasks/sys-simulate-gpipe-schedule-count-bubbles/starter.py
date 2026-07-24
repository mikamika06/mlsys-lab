def gpipe_schedule(p: int, m: int) -> dict:
    """Simulate GPipe's fill-drain pipeline schedule for p devices (stages)
    and m microbatches, assuming every forward and every backward task takes
    exactly 1 discrete integer time slot. See task.md for the exact
    dependency and ordering rules.

    Returns
    -------
    dict:
      timeline : list of p lists of (start, end) int tuples, one list per
        device, tasks in that device's execution order
        [F(0), F(1), ..., F(m-1), B(m-1), ..., B(0)].
      makespan : int -- the time slot at which the last task anywhere ends.
      bubble_slots : int -- total idle device-slots,
        p * makespan - p * 2 * m.
    """
    raise NotImplementedError('your code here')
