def simulate_active_set(arrival_iters: list[int], gen_lens: list[int], max_num_seqs: int) -> list[list[int]]:
    """Discrete-event continuous-batching simulation: each iteration,
    admit waiting requests (FIFO, up to max_num_seqs free slots), record
    the active set, decode one token per active request, then retire any
    request that has now reached its gen_len. Return a list of lists: the
    sorted active request indices at every iteration."""
    raise NotImplementedError('your code here')
