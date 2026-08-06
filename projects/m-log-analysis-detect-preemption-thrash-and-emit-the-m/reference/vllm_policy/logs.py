def detect_thrash(logs: list[dict], current_args: dict) -> dict | None:
    schedules = sum(1 for x in logs if x.get("event") == "schedule")
    preempts = sum(1 for x in logs if x.get("event") == "preempt")

    if schedules > 0 and (preempts / schedules) > 0.25:
        return {"max_num_seqs": max(1, current_args.get("max_num_seqs", 256) - 16)}
    return None


def classify_traces(traces: list[dict]) -> dict[str, str]:
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
