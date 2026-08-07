def generate_gpipe_schedule(num_stages: int, num_microbatches: int) -> list:
    events = []
    f_end = {}
    for m in range(num_microbatches):
        for p in range(num_stages):
            start = f_end.get((p, m - 1), 0) if m > 0 else 0
            if p > 0:
                prev_end = f_end[(p - 1, m)]
                start = max(start, prev_end)
            events.append({"stage": p, "mb": m, "type": "F", "start": start, "duration": 1})
            f_end[(p, m)] = start + 1

    b_end = {}
    last_f_end = f_end[(num_stages - 1, num_microbatches - 1)]
    for m in range(num_microbatches):
        for p in range(num_stages - 1, -1, -1):
            start = last_f_end if (m == 0 and p == num_stages - 1) else 0
            if m > 0:
                start = max(start, b_end.get((p, m - 1), 0))
            if p < num_stages - 1:
                start = max(start, b_end[(p + 1, m)])
            if p == num_stages - 1 and m == 0:
                start = max(start, f_end[(p, m)])
            events.append({"stage": p, "mb": m, "type": "B", "start": start, "duration": 1})
            b_end[(p, m)] = start + 1

    return events

def generate_1f1b_schedule(num_stages: int, num_microbatches: int) -> list:
    events = []
    stage_time = [0] * num_stages
    f_done = {}
    b_done = {}

    f_next = [0] * num_stages
    b_next = [0] * num_stages

    total_ops = 2 * num_stages * num_microbatches
    ops_done = 0

    while ops_done < total_ops:
        progress = False
        for p in range(num_stages):
            if b_next[p] < num_microbatches:
                m_b = b_next[p]
                dep_ok = (p == num_stages - 1) or ((p + 1, m_b) in b_done)
                f_ok = (p, m_b) in f_done
                ready_for_b = dep_ok and f_ok

                warmup_needed = num_stages - 1 - p
                must_warmup = f_next[p] < min(num_microbatches, warmup_needed + 1)
                can_do_f = f_next[p] < num_microbatches
                if p > 0 and can_do_f:
                    m_f = f_next[p]
                    if (p - 1, m_f) not in f_done:
                        can_do_f = False

                if ready_for_b and (not must_warmup or not can_do_f):
                    start = max(stage_time[p], b_done.get((p + 1, m_b), 0) if p < num_stages - 1 else 0)
                    events.append({"stage": p, "mb": m_b, "type": "B", "start": start, "duration": 1})
                    end = start + 1
                    stage_time[p] = end
                    b_done[(p, m_b)] = end
                    b_next[p] += 1
                    ops_done += 1
                    progress = True
                    continue

            if f_next[p] < num_microbatches:
                m_f = f_next[p]
                dep_ok = (p == 0) or ((p - 1, m_f) in f_done)
                if dep_ok:
                    start = max(stage_time[p], f_done.get((p - 1, m_f), 0) if p > 0 else 0)
                    events.append({"stage": p, "mb": m_f, "type": "F", "start": start, "duration": 1})
                    end = start + 1
                    stage_time[p] = end
                    f_done[(p, m_f)] = end
                    f_next[p] += 1
                    ops_done += 1
                    progress = True
                    continue
        if not progress:
            break
    return events

def generate_interleaved_1f1b_schedule(num_stages: int, num_microbatches: int, num_chunks: int) -> list:
    total_virtual_stages = num_stages * num_chunks
    events = []
    v_time = [0] * total_virtual_stages
    f_done = {}
    b_done = {}

    f_next = [0] * total_virtual_stages
    b_next = [0] * total_virtual_stages

    total_ops = 2 * total_virtual_stages * num_microbatches
    ops_done = 0

    while ops_done < total_ops:
        progress = False
        for rank in range(num_stages):
            v_stages = [rank + c * num_stages for c in range(num_chunks)]
            for v in v_stages:
                if b_next[v] < num_microbatches:
                    m = b_next[v]
                    dep_ok = (v == total_virtual_stages - 1) or ((v + 1, m) in b_done)
                    f_ok = (v, m) in f_done
                    if dep_ok and f_ok:
                        rank_time = max(v_time[r + c * num_stages] for c in range(num_chunks) for r in [rank])
                        dep_time = b_done.get((v + 1, m), 0) if v < total_virtual_stages - 1 else 0
                        start = max(rank_time, dep_time)
                        events.append({"stage": v, "mb": m, "type": "B", "start": start, "duration": 1})
                        end = start + 1
                        for c in range(num_chunks):
                            v_time[rank + c * num_stages] = max(v_time[rank + c * num_stages], end)
                        b_done[(v, m)] = end
                        b_next[v] += 1
                        ops_done += 1
                        progress = True
                        break

                if f_next[v] < num_microbatches:
                    m = f_next[v]
                    dep_ok = (v == 0) or ((v - 1, m) in f_done)
                    if dep_ok:
                        rank_time = max(v_time[r + c * num_stages] for c in range(num_chunks) for r in [rank])
                        dep_time = f_done.get((v - 1, m), 0) if v > 0 else 0
                        start = max(rank_time, dep_time)
                        events.append({"stage": v, "mb": m, "type": "F", "start": start, "duration": 1})
                        end = start + 1
                        for c in range(num_chunks):
                            v_time[rank + c * num_stages] = max(v_time[rank + c * num_stages], end)
                        f_done[(v, m)] = end
                        f_next[v] += 1
                        ops_done += 1
                        progress = True
                        break
        if not progress:
            break
    return events

def generate_zero_bubble_schedule(num_stages: int, num_microbatches: int) -> list:
    events = []
    stage_time = [0] * num_stages
    f_done = {}
    bw_done = {}
    bi_done = {}

    f_next = [0] * num_stages
    bw_next = [0] * num_stages
    bi_next = [0] * num_stages

    total_ops = 3 * num_stages * num_microbatches
    ops_done = 0

    while ops_done < total_ops:
        progress = False
        for p in range(num_stages):
            if bi_next[p] < num_microbatches:
                m = bi_next[p]
                dep_ok = (p == num_stages - 1) or ((p + 1, m) in bi_done)
                bw_ok = (p, m) in bw_done
                if dep_ok and bw_ok:
                    start = max(stage_time[p], bi_done.get((p + 1, m), 0) if p < num_stages - 1 else 0)
                    events.append({"stage": p, "mb": m, "type": "B_input", "start": start, "duration": 1})
                    end = start + 1
                    stage_time[p] = end
                    bi_done[(p, m)] = end
                    bi_next[p] += 1
                    ops_done += 1
                    progress = True
                    continue

            if bw_next[p] < num_microbatches:
                m = bw_next[p]
                f_ok = (p, m) in f_done
                if f_ok:
                    start = stage_time[p]
                    events.append({"stage": p, "mb": m, "type": "B_weight", "start": start, "duration": 1})
                    end = start + 1
                    stage_time[p] = end
                    bw_done[(p, m)] = end
                    bw_next[p] += 1
                    ops_done += 1
                    progress = True
                    continue

            if f_next[p] < num_microbatches:
                m = f_next[p]
                dep_ok = (p == 0) or ((p - 1, m) in f_done)
                if dep_ok:
                    start = max(stage_time[p], f_done.get((p - 1, m), 0) if p > 0 else 0)
                    events.append({"stage": p, "mb": m, "type": "F", "start": start, "duration": 1})
                    end = start + 1
                    stage_time[p] = end
                    f_done[(p, m)] = end
                    f_next[p] += 1
                    ops_done += 1
                    progress = True
                    continue
        if not progress:
            break
    return events
