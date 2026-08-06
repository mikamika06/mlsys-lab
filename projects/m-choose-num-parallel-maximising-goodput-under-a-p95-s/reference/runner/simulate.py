import numpy as np


def simulate_execution(num_parallel, requests, queue_capacity=10):
    """
    Simulate processing requests through slots and fixed queue.
    """
    running = []
    queue = []
    completed = []
    dropped = []

    req_map = {}
    for r in requests:
        req_map.setdefault(r["arrival_time"], []).append(r)

    timeline = sorted(list(req_map.keys()))

    events_queue = []
    for t in timeline:
        for r in req_map[t]:
            events_queue.append(r)

    events_queue.sort(key=lambda x: x["arrival_time"])

    active = []

    for req in events_queue:
        arr = float(req["arrival_time"])
        prefill = float(req["prefill_ms"])
        decode = float(req["decode_ms"])
        req_id = req["id"]

        active = [finish for finish in active if finish > arr]

        if len(active) < num_parallel:
            finish_time = arr + prefill + decode
            active.append(finish_time)
            active.sort()
            completed.append(
                {
                    "id": req_id,
                    "arrival_time": arr,
                    "start_time": arr,
                    "finish_time": finish_time,
                    "latency_ms": prefill + decode,
                    "wait_ms": 0.0,
                    "prefill_ms": prefill,
                }
            )
        else:
            earliest_available = active[0]
            if len(active) - num_parallel + len(queue) >= queue_capacity:
                dropped.append({"id": req_id, "arrival_time": arr, "reason": "queue_full"})
            else:
                start_time = max(arr, earliest_available)
                finish_time = start_time + prefill + decode
                active[0] = finish_time
                active.sort()
                completed.append(
                    {
                        "id": req_id,
                        "arrival_time": arr,
                        "start_time": start_time,
                        "finish_time": finish_time,
                        "latency_ms": finish_time - arr,
                        "wait_ms": start_time - arr,
                        "prefill_ms": prefill,
                    }
                )

    return {"completed": completed, "dropped": dropped}
