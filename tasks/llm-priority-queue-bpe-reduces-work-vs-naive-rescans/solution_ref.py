import heapq


def bpe_encode(ids, ranks):
    """Encode a symbol sequence with a BPE merge table using a priority queue.

    ``ids``   : list[int]           initial symbol ids (bytes 0..255 + merged ids).
    ``ranks`` : {(a, b): rank}      merging pair (a, b) yields id ``256 + rank``;
                                    lower rank = higher priority.

    Returns the fully merged sequence. Each merge only re-examines the two new
    neighbouring pairs, so the work does not scale with (length * #merges).
    """
    n = len(ids)
    if n < 2:
        return list(ids)

    # Doubly linked list over the symbol positions.
    sym = list(ids)                       # current symbol id at each node
    prev = [i - 1 for i in range(n)]      # prev[0] == -1
    nxt = [i + 1 for i in range(n)]
    nxt[n - 1] = -1                       # -1 == no neighbour
    alive = [True] * n

    # Seed the heap with every adjacent pair that has a rank: (rank, left_node).
    heap = []
    for i in range(n - 1):
        r = ranks.get((sym[i], sym[i + 1]))
        if r is not None:
            heapq.heappush(heap, (r, i))

    while heap:
        r, i = heapq.heappop(heap)
        if not alive[i]:
            continue
        j = nxt[i]
        if j == -1 or not alive[j]:
            continue
        # Stale entry: the neighbourhood changed since this pair was pushed.
        if ranks.get((sym[i], sym[j])) != r:
            continue

        # Merge node j into node i.
        sym[i] = 256 + r
        alive[j] = False
        k = nxt[j]
        nxt[i] = k
        if k != -1:
            prev[k] = i

        # Only the two freshly formed pairs can be new merge candidates.
        p = prev[i]
        if p != -1:
            rp = ranks.get((sym[p], sym[i]))
            if rp is not None:
                heapq.heappush(heap, (rp, p))
        if k != -1:
            rn = ranks.get((sym[i], sym[k]))
            if rn is not None:
                heapq.heappush(heap, (rn, i))

    # Walk the surviving nodes in order (nxt never points at a dead node).
    out = []
    i = 0
    while i != -1:
        out.append(sym[i])
        i = nxt[i]
    return out
