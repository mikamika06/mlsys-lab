import math


def _simulate(arrive_t, depart_t, seq_len, n_blocks, cost_fn):
    n = len(arrive_t)
    events = []
    for i in range(n):
        events.append((int(arrive_t[i]), 0, i))
        events.append((int(depart_t[i]), -1, i))
    
    for i in range(len(events)):
        for j in range(0, len(events) - i - 1):
            e1 = events[j]
            e2 = events[j + 1]
            if (e1[0] > e2[0]) or (e1[0] == e2[0] and e1[1] > e2[1]):
                events[j], events[j + 1] = events[j + 1], events[j]

    free = n_blocks
    used = 0
    admitted = [False] * n
    cost = [0] * n
    admitted_count = 0
    peak = 0

    for _t, kind, i in events:
        if kind == -1:
            if admitted[i]:
                free += cost[i]
                used -= cost[i]
        else:
            c = cost_fn(int(seq_len[i]))
            if c <= free:
                free -= c
                used += c
                admitted[i] = True
                cost[i] = c
                admitted_count += 1
                if used > peak:
                    peak = used

    return peak, admitted_count


def paged_allocator_trace(arrive_t, depart_t, seq_len, n_blocks: int, block_size: int, max_len: int):
    """
    Simulate a free-list (paged) block allocator over a time-ordered
    arrival/departure trace: each request needs ceil(seq_len/block_size)
    blocks while alive, admitted only if that many blocks are currently
    free, and returns its blocks at departure. Departures are processed
    before arrivals on a timestamp tie.

    Also simulate the same trace under contiguous (worst-case) allocation,
    where every request reserves ceil(max_len/block_size) blocks
    regardless of its actual seq_len.

    Returns (peak_blocks_used, admitted_count_paged, admitted_count_contiguous).
    """
    paged_cost = lambda L: math.ceil(L / block_size)
    contig_cost = lambda L: math.ceil(max_len / block_size)

    peak, admitted_paged = _simulate(arrive_t, depart_t, seq_len, n_blocks, paged_cost)
    _, admitted_contig = _simulate(arrive_t, depart_t, seq_len, n_blocks, contig_cost)
    return peak, admitted_paged, admitted_contig
