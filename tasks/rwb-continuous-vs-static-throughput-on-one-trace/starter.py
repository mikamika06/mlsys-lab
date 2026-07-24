def compare_batching_throughput(gen_lens, batch_size):
    """Compare static vs continuous batching makespan and throughput on one trace.

    gen_lens: list of N positive ints, generation length per request, in
        arrival order (both policies process this exact order).
    batch_size: number of concurrent slots (also the static batch size).

    Returns (makespan_static, makespan_cont, throughput_ratio):
      makespan_static: sum over consecutive batches of `batch_size`
        requests (arrival order, last batch may be smaller) of the max
        length within each batch (batches run strictly sequentially).
      makespan_cont: max completion time under work-conserving
        continuous batching with `batch_size` slots (each arriving
        request goes to whichever slot is free soonest).
      throughput_ratio: (sum(gen_lens)/makespan_cont) /
        (sum(gen_lens)/makespan_static), equivalently
        makespan_static / makespan_cont.
    """
    raise NotImplementedError('your code here')
