"""Milestone 1 checker: Thread scaling sweep and oversubscription analysis."""

import ref


def check(workdir):
    from scaling.threads import analyze_thread_sweep, find_oversubscription_point

    out = {"latency_ratio": 0.0, "oversubscription_matched": 0.0}

    ref_point = ref.find_oversubscription_point(ref.TOPOLOGY, ref.SWEEP_LATENCIES)
    got_point = find_oversubscription_point(ref.TOPOLOGY, ref.SWEEP_LATENCIES)

    if got_point == ref_point:
        out["oversubscription_matched"] = 1.0
    else:
        out["_note"] = f"oversubscription point: expected {ref_point}, got {got_point}"

    ref_sweep = ref.analyze_thread_sweep(ref.SWEEP_LATENCIES, ref.WORK_UNITS)
    got_sweep = analyze_thread_sweep(ref.SWEEP_LATENCIES, ref.WORK_UNITS)

    correct_entries = 0
    total_entries = len(ref_sweep)

    for k, ref_val in ref_sweep.items():
        if k in got_sweep:
            g_val = got_sweep[k]
            tp_diff = abs(g_val.get("throughput", 0) - ref_val["throughput"])
            lr_diff = abs(g_val.get("latency_ratio", 0) - ref_val["latency_ratio"])
            if tp_diff < 1e-3 and lr_diff < 1e-3:
                correct_entries += 1

    ratio = float(correct_entries) / float(total_entries) if total_entries > 0 else 0.0
    out["latency_ratio"] = ratio

    return out
