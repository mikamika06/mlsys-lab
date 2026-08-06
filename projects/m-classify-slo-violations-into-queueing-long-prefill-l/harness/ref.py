import random


def get_data():
    random.seed(42)
    reqs = []
    for i in range(50):
        arr = i * 0.1
        start = arr + random.uniform(0.01, 0.5)
        pref = start + random.uniform(0.05, 0.8)
        fin = pref + random.uniform(0.1, 1.5)
        reqs.append({
            "id": f"req_{i}",
            "arrival_time": arr,
            "start_time": start,
            "prefill_end_time": pref,
            "finish_time": fin
        })
    return reqs, 1.2


def identify_violations(requests, slo):
    res = []
    for r in requests:
        e2e = r["finish_time"] - r["arrival_time"]
        if e2e > slo:
            res.append(r["id"])
    return res


def classify_violations(requests, slo):
    out = {}
    for r in requests:
        e2e = r["finish_time"] - r["arrival_time"]
        if e2e <= slo:
            continue
        q = r["start_time"] - r["arrival_time"]
        p = r["prefill_end_time"] - r["start_time"]
        gen = r["finish_time"] - r["prefill_end_time"]
        mx = max(q, p, gen)
        if mx == q:
            out[r["id"]] = "queueing"
        elif mx == p:
            out[r["id"]] = "long_prefill"
        else:
            out[r["id"]] = "long_output"
    return out
