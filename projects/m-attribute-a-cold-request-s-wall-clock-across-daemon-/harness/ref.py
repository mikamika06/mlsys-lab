PHASE_NAMES = ("daemon", "load", "prefill", "decode")

CASES = [
    {"events": [
        {"name": "request_in", "t": 1000},
        {"name": "daemon_start", "t": 1002},
        {"name": "daemon_end", "t": 1900},
        {"name": "load_start", "t": 1900},
        {"name": "load_end", "t": 5100},
        {"name": "prefill_start", "t": 5100},
        {"name": "prefill_end", "t": 5680},
        {"name": "decode_start", "t": 5680},
        {"name": "decode_end", "t": 6820},
        {"name": "request_out", "t": 6830},
    ]},
    {"events": [
        {"name": "request_in", "t": 0},
        {"name": "load_start", "t": 4},
        {"name": "load_end", "t": 2504},
        {"name": "prefill_start", "t": 2504},
        {"name": "prefill_end", "t": 2900},
        {"name": "decode_start", "t": 2900},
        {"name": "decode_end", "t": 3400},
        {"name": "request_out", "t": 3410},
    ]},
    {"events": [
        {"name": "request_in", "t": 500},
        {"name": "prefill_start", "t": 503},
        {"name": "prefill_end", "t": 560},
        {"name": "decode_start", "t": 560},
        {"name": "decode_end", "t": 810},
        {"name": "request_out", "t": 815},
    ]},
    {"events": [
        {"name": "decode_end", "t": 9000},
        {"name": "request_in", "t": 2000},
        {"name": "prefill_end", "t": 7800},
        {"name": "daemon_end", "t": 2600},
        {"name": "load_end", "t": 7500},
        {"name": "decode_start", "t": 7800},
        {"name": "prefill_start", "t": 7500},
        {"name": "load_start", "t": 2600},
        {"name": "daemon_start", "t": 2003},
        {"name": "request_out", "t": 9020},
    ]},
    {"events": [
        {"name": "request_in", "t": 0},
        {"name": "daemon_start", "t": 2},
        {"name": "daemon_end", "t": 2},
        {"name": "load_start", "t": 2},
        {"name": "load_end", "t": 1502},
        {"name": "prefill_start", "t": 1502},
        {"name": "prefill_end", "t": 1550},
        {"name": "decode_start", "t": 1550},
        {"name": "decode_end", "t": 1900},
        {"name": "request_out", "t": 1905},
    ]},
]


def build_phases(events):
    marks = {}
    for e in events:
        marks[e["name"]] = e["t"]
    phases = []
    for name in PHASE_NAMES:
        start_key = name + "_start"
        end_key = name + "_end"
        if start_key in marks and end_key in marks:
            start = marks[start_key]
            end = marks[end_key]
            phases.append({
                "name": name,
                "start": start,
                "end": end,
                "duration_ms": end - start,
            })
    return phases


def total_wall_clock(events):
    marks = {e["name"]: e["t"] for e in events}
    return marks["request_out"] - marks["request_in"]


def phase_breakdown(events):
    return {p["name"]: p["duration_ms"] for p in build_phases(events)}


def classify_request(events):
    names = {p["name"] for p in build_phases(events)}
    if "daemon" in names:
        return "cold"
    if "load" in names:
        return "warm_daemon"
    return "hot"


def unattributed_ms(events):
    total = total_wall_clock(events)
    attributed = sum(p["duration_ms"] for p in build_phases(events))
    return total - attributed


def cumulative_ms(events, checkpoints):
    t0 = next(e["t"] for e in events if e["name"] == "request_in")
    phases = build_phases(events)
    out = []
    for offset in checkpoints:
        moment = t0 + offset
        total = 0
        for p in phases:
            span = min(moment, p["end"]) - p["start"]
            if span < 0:
                span = 0
            elif span > p["duration_ms"]:
                span = p["duration_ms"]
            total += span
        out.append(total)
    return out


def checkpoints_for(events):
    total = total_wall_clock(events)
    n = 9
    step = (total + 50) / n
    return [round(i * step) for i in range(n + 1)]
