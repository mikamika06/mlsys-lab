def _makespan(reqs, slots):
    available = [0.0] * slots
    for r in sorted(reqs, key=lambda x: x["arrival"]):
        idx = min(range(slots), key=lambda i: available[i])
        start = max(float(r["arrival"]), available[idx])
        available[idx] = start + float(r["prefill"]) + float(r["decode"])
    return max(available) if available else 0.0


def minimum_max_num_seqs(reqs, target_makespan):
    for s in range(1, len(reqs) + 1):
        if _makespan(reqs, s) <= target_makespan:
            return s
    return -1
