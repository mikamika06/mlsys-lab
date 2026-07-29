from .phases import build_phases


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
