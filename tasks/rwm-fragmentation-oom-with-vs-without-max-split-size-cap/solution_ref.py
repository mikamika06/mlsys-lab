def replay_trace(trace, capacity, max_split_size=None):
    free_starts = {}   # start -> size
    free_ends = {}      # end -> start
    live = {}            # name -> (start, size)
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
                if sz >= size and (max_split_size is None or sz <= max_split_size or sz == size):
                    cands.append((sz, s))
            if cands:
                cands.sort()
                sz, s = cands[0]
                end = s + sz
                del free_starts[s]
                del free_ends[end]
                if sz > size:
                    add_free(s + size, sz - size)
                live[name] = (s, size)
                peak = max(peak, reserved)
            else:
                if reserved + size > capacity:
                    oom = True
                    continue
                start = reserved
                reserved += size
                live[name] = (start, size)
                peak = max(peak, reserved)
        else:
            _, name = op
            s, sz = live.pop(name)
            add_free(s, sz)

    return {"oom": oom, "peak_reserved": peak}
