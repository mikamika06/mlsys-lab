def parse_artifact(data):
    return [
        {
            "id": t["id"],
            "size": t["size"],
            "constant": t["constant"],
            "start": t["start"],
            "end": t["end"]
        }
        for t in data["tensors"]
    ]


def find_peak(tensors):
    max_time = max((t["end"] for t in tensors), default=0)
    peak = 0
    peak_tensors = []
    for time in range(max_time + 1):
        active = [t for t in tensors if t["start"] <= time < t["end"]]
        current = sum(t["size"] for t in active)
        if current > peak:
            peak = current
            peak_tensors = [t["id"] for t in active]
    return peak, peak_tensors


def split_program_data(tensors):
    constants = [t for t in tensors if t["constant"]]
    activations = [t for t in tensors if not t["constant"]]
    return constants, activations


def replan_buffers(activations):
    acts = sorted(activations, key=lambda x: (x["start"], -x["size"]))
    allocations = {}
    for t in acts:
        offset = 0
        while True:
            overlap = False
            for other_id, other_off in allocations.items():
                other = next(x for x in activations if x["id"] == other_id)
                l_overlap = (t["start"] < other["end"]) and (other["start"] < t["end"])
                if l_overlap:
                    m_overlap = not (offset + t["size"] <= other_off or other_off + other["size"] <= offset)
                    if m_overlap:
                        overlap = True
                        offset = other_off + other["size"]
                        break
            if not overlap:
                allocations[t["id"]] = offset
                break
    return allocations


def check_budget(tensors, budget):
    constants, activations = split_program_data(tensors)
    const_size = sum(t["size"] for t in constants)
    allocs = replan_buffers(activations)
    if not allocs:
        peak_act = 0
    else:
        peak_act = max(allocs[t["id"]] + t["size"] for t in activations)
    return const_size + peak_act <= budget
