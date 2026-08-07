import numpy as np

TRACES = [
    {
        "traceEvents": [
            {"name": "aten::linear", "ts": 10.0, "dur": 120.0, "args": {}},
            {"name": "PagedAdamW8bit::step", "ts": 150.0, "dur": 300.0, "args": {"page_faults": 10, "bytes_transferred": 1024, "step": 0}},
            {"name": "PagedAdamW8bit::step", "ts": 500.0, "dur": 2500.0, "args": {"page_faults": 2, "bytes_transferred": 8192, "step": 1}},
            {"name": "PagedAdamW8bit::step", "ts": 3100.0, "dur": 400.0, "args": {"page_faults": 8, "bytes_transferred": 2048, "step": 2}},
        ]
    },
    {
        "traceEvents": [
            {"name": "optimizer_paged_adamw", "cat": "paged_adamw", "ts": 20.0, "dur": 800.0, "args": {"page_faults": 1, "step": 10}},
            {"name": "optimizer_paged_adamw", "cat": "paged_adamw", "ts": 900.0, "dur": 100.0, "args": {"page_faults": 20, "step": 11}},
        ]
    }
]


def parse_trace_events(trace_data):
    raw_events = trace_data.get("traceEvents", trace_data if isinstance(trace_data, list) else [])
    parsed = []
    for ev in raw_events:
        if not isinstance(ev, dict):
            continue
        name = str(ev.get("name", ""))
        cat = str(ev.get("cat", ""))
        args = ev.get("args", {})
        if "PagedAdamW8bit" in name or "paged_adamw" in cat or args.get("is_paged", False):
            parsed.append({
                "name": name,
                "ts": float(ev.get("ts", 0.0)),
                "dur": float(ev.get("dur", 0.0)),
                "page_faults": int(args.get("page_faults", 0)),
                "bytes_transferred": int(args.get("bytes_transferred", 0)),
                "step": int(args.get("step", 0))
            })
    return parsed


def find_spillover_spike(events):
    if not events:
        return {"argmin_index": -1, "max_ratio": 0.0}
    durations = np.array([e["dur"] for e in events], dtype=np.float64)
    faults = np.array([max(1, e["page_faults"]) for e in events], dtype=np.float64)
    ratios = durations / faults
    neg_ratios = -ratios
    target_idx = int(np.argmin(neg_ratios))
    return {
        "argmin_index": target_idx,
        "max_ratio": float(ratios[target_idx])
    }
