import json

def generate_pte_artifact():
    return {
        "version": 1,
        "tensors": [
            {"id": 0, "name": "weight_0", "size": 1024, "is_weight": True},
            {"id": 1, "name": "weight_1", "size": 2048, "is_weight": True},
            {"id": 2, "name": "act_in", "size": 512, "is_weight": False},
            {"id": 3, "name": "act_temp1", "size": 4096, "is_weight": False},
            {"id": 4, "name": "act_temp2", "size": 2048, "is_weight": False},
            {"id": 5, "name": "act_out", "size": 512, "is_weight": False}
        ],
        "operators": [
            {"name": "linear_1", "inputs": [2, 0], "outputs": [3], "start": 0, "end": 2},
            {"name": "relu_1", "inputs": [3], "outputs": [4], "start": 2, "end": 3},
            {"name": "linear_2", "inputs": [4, 1], "outputs": [5], "start": 3, "end": 5}
        ]
    }

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

def expected_peak_and_source(pte_data):
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

def expected_separation(pte_data):
    weights = [t for t in pte_data["tensors"] if t["is_weight"]]
    activations = [t for t in pte_data["tensors"] if not t["is_weight"]]
    weight_size = sum(w["size"] for w in weights)
    activation_size = max(t["size"] for t in activations) if activations else 0
    return weight_size, activation_size

def expected_plan(pte_data):
    from pte_plan.planner import plan_buffers
    return plan_buffers(pte_data)

def get_device_budget(pte_data):
    return 6000
