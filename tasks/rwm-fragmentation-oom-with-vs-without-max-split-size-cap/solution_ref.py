def replay_trace(trace, capacity, max_split_size=None):
    free_starts = {}
    free_ends = {}
    live = {}
    reserved = 0
    peak = 0
    oom = False

    def add_free(start, size):
        end = start + size
        if end in free_starts:
            rsize = free_starts.pop(end)
            rend = end + rsize
            del free_ends[rend]
            size += rsize
            end = start + size
        if start in free_ends:
            lstart = free_ends.pop(start)
            lsize = free_starts.pop(lstart)
            size += lsize
            start = lstart
            end = start + size
        free_starts[start] = size
        free_ends[end] = start

    for op in trace:
        if oom:
            break
        if op[0] == "alloc":
            _, name, size = op
            cands = []
            for s, sz in free_starts.items():
                if sz >= size:
                    if max_split_size is None or sz <= max_split_size or sz == size:
                        cands.append((sz, s))
            if cands:
                best_sz, best_s = cands[0]
                for i in range(1, len(cands)):
                    curr_sz, curr_s = cands[i]
                    if curr_sz < best_sz:
                        best_sz = curr_sz
                        best_s = curr_s
                    elif curr_sz == best_sz:
                        if curr_s < best_s:
                            best_s = curr_s
                sz = best_sz
                s = best_s
                end = s + sz
                del free_starts[s]
                del free_ends[end]
                if sz > size:
                    add_free(s + size, sz - size)
                live[name] = (s, size)
                if reserved > peak:
                    peak = reserved
            else:
                if reserved + size > capacity:
                    oom = True
                    continue
                start = reserved
                reserved += size
                live[name] = (start, size)
                if reserved > peak:
                    peak = reserved
        else:
            _, name = op
            s, sz = live.pop(name)
            add_free(s, sz)

    return {"oom": oom, "peak_reserved": peak}
