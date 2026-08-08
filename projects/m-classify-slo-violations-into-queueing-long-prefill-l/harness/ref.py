import random

random.seed(42)
REQUESTS = []
for i in range(20):
    arrival = i * 5
    start = arrival + random.choice([2, 20, 50])
    prompt_tokens = random.randint(10, 200)
    prefill_time = prompt_tokens * 0.4
    output_tokens = random.randint(5, 100)
    output_time = output_tokens * 0.8
    finish = start + prefill_time + output_time
    REQUESTS.append({
        "request_id": i,
        "arrival_time": float(arrival),
        "start_time": float(start),
        "finish_time": float(finish),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "prefill_time": float(prefill_time)
    })
SLO_TARGET = 100.0

def parse_request(raw):
    arrival = raw["arrival_time"]
    start = raw["start_time"]
    finish = raw["finish_time"]
    prompt_tokens = raw["prompt_tokens"]
    output_tokens = raw["output_tokens"]
    queue_time = start - arrival
    prefill_time = raw.get("prefill_time", prompt_tokens * 0.5)
    output_time = finish - (start + prefill_time)
    total_latency = finish - arrival
    return {
        "request_id": raw["request_id"],
        "queue_time": queue_time,
        "prefill_time": prefill_time,
        "output_time": output_time,
        "total_latency": total_latency,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens
    }

def classify_violation(req, slo_target):
    if req["total_latency"] <= slo_target:
        return "none"
    qt = req["queue_time"]
    pt = req["prefill_time"]
    ot = req["output_time"]
    if qt >= pt and qt >= ot:
        return "queueing"
    elif pt >= ot:
        return "long-prefill"
    else:
        return "long-output"

def generate_report(requests, slo_target):
    counts = {"none": 0, "queueing": 0, "long-prefill": 0, "long-output": 0}
    for r in requests:
        parsed = parse_request(r)
        cause = classify_violation(parsed, slo_target)
        counts[cause] += 1
    return counts
