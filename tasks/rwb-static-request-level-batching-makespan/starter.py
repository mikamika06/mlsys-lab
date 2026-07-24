import numpy as np


def static_batching_makespan(output_lens: np.ndarray, batch_size: int):
    """
    Simulate STATIC (request-level) batching: batches of up to `batch_size`
    requests are formed FCFS from `output_lens` (in order); a batch runs
    for max(output_lens in that batch) iterations (shorter requests idle
    until the longest finishes); the next batch cannot start until the
    whole previous batch has drained.

    Return (makespan, batch_iter_counts): total iterations across all
    batches, and the per-batch iteration count list. See task.md.
    """
    raise NotImplementedError('your code here')
