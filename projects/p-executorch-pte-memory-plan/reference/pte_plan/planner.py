def get_tensor_intervals(pte_data):
    intervals = {}
    for t in pte_data["tensors"]:
        if t["is_weight"]:
            intervals[t["id"]] = (0, float("inf"))
        else:
            start = float("inf")
            end = -1
            for op in pte_data["operators"]:
                if t["id"] in op["inputs"] or t["id"] in op["outputs"]:
                    start = min(start, op["start"])
                    end = max(end, op["end"])
            if start == float("inf"):
                start = 0
                end = 0
            intervals[t["id"]] = (start, end)
    return intervals

def plan_buffers(pte_data):
    intervals = get_tensor_intervals(pte_data)
    activations = [t for t in pte_data["tensors"] if not t["is_weight"]]
    activations.sort(key=lambda x: intervals[x["id"]][0])

    allocated_buffers = {}
    buffer_end_times = []
    current_offset = 0
    max_offset = 0

    for t in activations:
        tid = t["id"]
        start, end = intervals[tid]
        size = t["size"]

        reused_offset = None
        for i, (b_end, b_offset, b_size) in enumerate(buffer_end_times):
            if b_end <= start and b_size >= size:
                reused_offset = b_offset
                buffer_end_times[i] = (end, b_offset, b_size)
                break

        if reused_offset is not None:
            allocated_buffers[tid] = reused_offset
        else:
            allocated_buffers[tid] = current_offset
            buffer_end_times.append((end, current_offset, size))
            current_offset += size
            max_offset = max(max_offset, current_offset)

    return max_offset, allocated_buffers
