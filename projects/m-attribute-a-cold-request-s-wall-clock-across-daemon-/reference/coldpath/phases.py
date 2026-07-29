PHASE_NAMES = ("daemon", "load", "prefill", "decode")


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
