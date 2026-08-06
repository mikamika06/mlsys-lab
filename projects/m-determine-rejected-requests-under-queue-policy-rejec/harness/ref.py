SCENARIOS = [
    {"queue_size": 50, "max_queue_size": 100, "incoming_count": 60, "policy": "REJECT", "want_rejected": 10},
    {"queue_size": 90, "max_queue_size": 100, "incoming_count": 20, "policy": "REJECT", "want_rejected": 10},
    {"queue_size": 40, "max_queue_size": 100, "incoming_count": 30, "policy": "REJECT", "want_rejected": 0},
]

ATTRIBUTIONS = [
    {"b_wait": 5.0, "b_exec": 15.0, "c_wait": 40.0, "c_exec": 16.0, "want": "batcher"},
    {"b_wait": 5.0, "b_exec": 15.0, "c_wait": 6.0, "c_exec": 50.0, "want": "model"},
    {"b_wait": 10.0, "b_exec": 20.0, "c_wait": 11.0, "c_exec": 21.0, "want": "mixed"},
]

def compute_rejected_requests(queue_size, max_queue_size, incoming_count, policy):
    if policy != "REJECT":
        return 0
    space = max(0, max_queue_size - queue_size)
    return max(0, incoming_count - space)

def attribute_latency(baseline_wait, baseline_exec, current_wait, current_exec):
    w_diff = current_wait - baseline_wait
    e_diff = current_exec - baseline_exec
    if w_diff > e_diff and w_diff > 0:
        return "batcher"
    if e_diff > w_diff and e_diff > 0:
        return "model"
    return "mixed"
