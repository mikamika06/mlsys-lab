def schedule_admit_preempt(arrivals, lengths, priorities, budget, num_steps):
    """Simulate a preemptive-priority continuous-batching scheduler with a
    fixed running-set capacity `budget` over `num_steps` steps.

    `arrivals`, `lengths`, `priorities` are same-length lists of ints, one
    entry per request (request id == its index). Lower `priorities` value
    means higher priority.

    Returns a list of length `num_steps`. Step `t`'s entry is a dict:
      {"admitted": [...], "preempted": [...]}
    - "admitted": sorted request ids newly in the running set at step `t`
      that were not running at step `t-1`.
    - "preempted": sorted request ids that were running at step `t-1`,
      still have work left, but are not in the running set at step `t`.

    See task.md for the exact scheduling rule.
    """
    raise NotImplementedError('your code here')
