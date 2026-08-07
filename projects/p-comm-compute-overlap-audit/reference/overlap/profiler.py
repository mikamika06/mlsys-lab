def collect_profile(events):
    timeline = []
    for ev in events:
        timeline.append({
            "name": ev.get("name"),
            "start": ev.get("start"),
            "dur": ev.get("dur"),
            "cat": ev.get("cat", "compute")
        })
    return timeline

def identify_communication(profile):
    comm_events = [ev for ev in profile if "all_reduce" in ev["name"] or ev.get("cat") == "comm"]
    total_comm_dur = sum(ev["dur"] for ev in comm_events)
    return {"comm_events": comm_events, "total_duration": total_comm_dur}
