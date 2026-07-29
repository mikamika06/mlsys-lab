from .phases import build_phases


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
