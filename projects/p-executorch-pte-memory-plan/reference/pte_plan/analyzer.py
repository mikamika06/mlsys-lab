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

def get_peak_memory(pte_data):
    intervals = get_tensor_intervals(pte_data)
    max_t = max(op["end"] for op in pte_data["operators"]) if pte_data["operators"] else 0
    peak = 0
    for step in range(max_t + 1):
        current = 0
        for t in pte_data["tensors"]:
            start, end = intervals[t["id"]]
            if start <= step < end or (start == end == step):
                current += t["size"]
        if current > peak:
            peak = current
    transients = [t for t in pte_data["tensors"] if not t["is_weight"]]
    peak_tensor = max(transients, key=lambda x: x["size"])["name"] if transients else pte_data["tensors"][0]["name"]
    return peak, peak_tensor

def separate_program_and_data(pte_data):
    weights = [t for t in pte_data["tensors"] if t["is_weight"]]
    activations = [t for t in pte_data["tensors"] if not t["is_weight"]]
    weight_size = sum(w["size"] for w in weights)
    activation_size = max(t["size"] for t in activations) if activations else 0
    return weight_size, activation_size
