def admit(queue, active, max_total_tokens, max_prefill_tokens, waiting_served_ratio):
    if not queue:
        return []
    if active and len(queue) <= waiting_served_ratio * len(active):
        return []
    admitted_ids = []
    prefill_sum = 0
    active_sum = sum(req["input_len"] + req["generated_len"] for req in active)
    for req in queue:
        if prefill_sum + req["input_len"] > max_prefill_tokens:
            break
        if active_sum + prefill_sum + req["input_len"] > max_total_tokens:
            break
        admitted_ids.append(req["id"])
        prefill_sum += req["input_len"]
    return admitted_ids

def compute_utilization(log, max_total_tokens):
    if not log:
        return 0.0
    total_util = 0.0
    for tick in log:
        active_sum = sum(req["input_len"] + req["generated_len"] for req in tick["active"])
        prefill_sum = sum(req["input_len"] for req in tick["prefill"])
        total_util += (active_sum + prefill_sum) / max_total_tokens
    return total_util / len(log)

M1_CASES = [
    ([{"id": "1", "input_len": 10}], [], 100, 100, 1.2),
    ([{"id": "1", "input_len": 10}], [{"id": "2", "input_len": 10, "generated_len": 5}], 100, 100, 1.2),
    ([{"id": "1", "input_len": 10}, {"id": "2", "input_len": 10}], [{"id": "3", "input_len": 10, "generated_len": 5}], 100, 100, 1.2),
    ([{"id": "1", "input_len": 50}, {"id": "2", "input_len": 50}], [], 80, 100, 1.2),
    ([{"id": "1", "input_len": 50}, {"id": "2", "input_len": 50}], [], 100, 80, 1.2),
    ([{"id": "1", "input_len": 10}], [{"id": "2", "input_len": 40, "generated_len": 50}], 90, 100, 0.0),
]

M2_LOGS = [
    (
        [
            {"active": [], "prefill": [{"input_len": 10}]},
            {"active": [{"input_len": 10, "generated_len": 1}], "prefill": []},
            {"active": [{"input_len": 10, "generated_len": 2}], "prefill": [{"input_len": 20}]},
        ],
        100
    ),
    (
        [
            {"active": [{"input_len": 50, "generated_len": 10}], "prefill": [{"input_len": 40}]},
            {"active": [{"input_len": 50, "generated_len": 11}, {"input_len": 40, "generated_len": 1}], "prefill": []},
        ],
        200
    )
]
