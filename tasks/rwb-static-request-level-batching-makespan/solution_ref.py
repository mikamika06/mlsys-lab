import numpy as np


def static_batching_makespan(output_lens: np.ndarray, batch_size: int):
    """
    output_lens : 1-D int array, the number of decode iterations each
        queued request needs, in ARRIVAL (queue) order.
    batch_size  : S, the max number of requests processed together in one
        static batch.

    STATIC (request-level) batching: form a batch from the next up-to-S
    queued requests (FCFS, in order). Run that batch for
    max(output_lens in batch) iterations -- any request that finishes
    sooner just idles (its slot is wasted) until the longest request in
    the batch completes. Only THEN does the next batch of up to S
    requests start (batch-blocking: a new batch cannot start until the
    whole previous batch has fully drained).

    Returns (makespan, batch_iter_counts):
      makespan          -- total iterations across ALL batches (int): the
                            sum of every batch's max(output_lens in batch).
      batch_iter_counts -- list of ints, one per batch in order: that
                            batch's iteration count (== its max output_len).
    """
    output_lens = np.asarray(output_lens, dtype=np.int64)
    counts = []
    for i in range(0, len(output_lens), batch_size):
        chunk = output_lens[i:i + batch_size]
        counts.append(int(np.max(chunk)))
    makespan = int(sum(counts))
    return makespan, counts
