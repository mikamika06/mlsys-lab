import random


def detect_thrash(logs, current_args):
    schedules = sum(1 for x in logs if x.get("event") == "schedule")
    preempts = sum(1 for x in logs if x.get("event") == "preempt")
    if schedules > 0 and preempts / schedules > 0.25:
        return {"max_num_seqs": max(1, current_args.get("max_num_seqs", 256) - 16)}
    return None


def classify_traces(traces):
    out = {}
    for t in traces:
        if not t["series"]:
            out[t["id"]] = "idle"
            continue
        q_avg = sum(p["q"] for p in t["series"]) / len(t["series"])
        kv_avg = sum(p["kv"] for p in t["series"]) / len(t["series"])
        if q_avg > 10.0:
            if kv_avg > 0.85:
                out[t["id"]] = "capacity-bound"
            else:
                out[t["id"]] = "arrival-bound"
        else:
            out[t["id"]] = "idle"
    return out


def simulate(requests, age_factor):
    pending = {r["id"]: dict(r) for r in requests}
    t = 0
    completions = {}
    while pending:
        available = [r for r in pending.values() if r["arrival"] <= t]
        if not available:
            t += 1
            continue
        best = None
        best_score = -1e9
        for r in available:
            score = r["prio"] + (t - r["arrival"]) * age_factor
            if score > best_score or (score == best_score and (best is None or r["id"] < best["id"])):
                best = r
                best_score = score
        best["work"] -= 1
        if best["work"] == 0:
            completions[best["id"]] = t + 1
            del pending[best["id"]]
        t += 1
    return completions


def _gen_logs():
    rng = random.Random(42)
    out = []
    for _ in range(5):
        log = [{"event": "schedule", "req": i} for i in range(100)]
        log += [{"event": "preempt", "req": i} for i in range(10)]
        rng.shuffle(log)
        out.append((log, {"max_num_seqs": 256}))

        log_thrash = [{"event": "schedule", "req": i} for i in range(100)]
        log_thrash += [{"event": "preempt", "req": i} for i in range(30)]
        rng.shuffle(log_thrash)
        out.append((log_thrash, {"max_num_seqs": 256}))
    return out


def _gen_traces():
    rng = random.Random(43)
    out = []
    for i in range(10):
        if i < 3:
            series = [{"q": rng.uniform(0, 5), "kv": rng.uniform(0.1, 0.5)} for _ in range(20)]
        elif i < 7:
            series = [{"q": rng.uniform(15, 30), "kv": rng.uniform(0.86, 0.99)} for _ in range(20)]
        else:
            series = [{"q": rng.uniform(15, 30), "kv": rng.uniform(0.2, 0.6)} for _ in range(20)]
        out.append({"id": f"t_{i}", "series": series})
    out.append({"id": "empty", "series": []})
    return out


LOG_CASES = _gen_logs()
TRACE_CASES = _gen_traces()
REQ_CASES = [
    [
        {"id": 4, "arrival": 0, "prio": 0, "work": 2},
        {"id": 1, "arrival": 1, "prio": 10, "work": 2},
        {"id": 2, "arrival": 3, "prio": 10, "work": 2},
        {"id": 3, "arrival": 5, "prio": 10, "work": 2}
    ],
    [
        {"id": i, "arrival": i * 2, "prio": 50 - i, "work": 3} for i in range(10)
    ]
]
