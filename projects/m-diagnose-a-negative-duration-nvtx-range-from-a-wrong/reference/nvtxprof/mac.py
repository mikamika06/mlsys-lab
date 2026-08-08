def analyze_mac_trace(trace_events, target_phases):
    phase_totals = {p: 0.0 for p in target_phases}
    phase_selves = {p: 0.0 for p in target_phases}

    def contains(outer, inner):
        o_start, o_end = outer["ts"], outer["ts"] + outer["dur"]
        i_start, i_end = inner["ts"], inner["ts"] + inner["dur"]
        return o_start <= i_start and i_end <= o_end

    for evt in trace_events:
        name = evt.get("name")
        if name not in target_phases:
            continue

        dur = float(evt.get("dur", 0.0))
        tid = evt.get("tid")
        phase_totals[name] += dur

        descendants = []
        for other in trace_events:
            if other is evt or other.get("tid") != tid:
                continue
            if contains(evt, other):
                descendants.append(other)

        immediate_children = []
        for d in descendants:
            is_nested = False
            for parent_candidate in descendants:
                if parent_candidate is d:
                    continue
                if contains(parent_candidate, d):
                    p_dur = parent_candidate.get("dur", 0.0)
                    d_dur = d.get("dur", 0.0)
                    if p_dur > d_dur or (
                        p_dur == d_dur
                        and trace_events.index(parent_candidate)
                        < trace_events.index(d)
                    ):
                        is_nested = True
                        break
            if not is_nested:
                immediate_children.append(d)

        child_dur_sum = sum(
            float(c.get("dur", 0.0)) for c in immediate_children
        )
        self_time = max(0.0, dur - child_dur_sum)
        phase_selves[name] += self_time

    rankings = sorted(target_phases, key=lambda p: (-phase_selves[p], p))

    phase_metrics = {}
    for rank, p in enumerate(rankings, 1):
        phase_metrics[p] = {
            "total_time": phase_totals[p],
            "self_time": phase_selves[p],
            "rank": rank,
        }

    return {"rankings": rankings, "phase_metrics": phase_metrics}
