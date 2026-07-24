def paged_allocator_trace(arrive_t, depart_t, seq_len, n_blocks: int, block_size: int, max_len: int):
    """Simulate a paged block allocator over an arrival/free trace, and
    compare admission against contiguous (worst-case) allocation.

    arrive_t, depart_t, seq_len: equal-length sequences, one entry per
        request: arrival timestamp, departure timestamp, and the number
        of tokens of KV cache it needs while alive.
    n_blocks: total physical block pool size.
    block_size: tokens per block.
    max_len: the worst-case context length a contiguous allocator must
        reserve room for, per request, regardless of actual seq_len.

    Build a combined, time-ordered event list (arrivals and departures);
    on a timestamp tie, process departures before arrivals (freed
    capacity is available to same-instant arrivals). Maintain a simple
    free-block counter (blocks are interchangeable -- no physical
    contiguity is needed for a paged allocator):

      - PAGED: at arrival, a request needs ceil(seq_len / block_size)
        blocks; admit it (deduct from the free count) only if that many
        are currently free, else reject it (it gets nothing, and its
        departure event does nothing). At an admitted request's
        departure, return its blocks to the free count. Track the peak
        number of blocks in use at any point across the whole trace.
      - CONTIGUOUS: identical simulation, except every request's cost is
        the fixed ceil(max_len / block_size), regardless of its actual
        seq_len.

    Returns (peak_blocks_used, admitted_count_paged, admitted_count_contiguous)
    -- peak_blocks_used is from the PAGED simulation.
    """
    raise NotImplementedError('your code here')
