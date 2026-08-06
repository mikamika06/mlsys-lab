def diagnose_range(events):
    pushes = {}
    for ev in events:
        if ev.get("ph") == "B":
            pushes[ev["name"]] = ev
        elif ev.get("ph") == "E":
            if ev["name"] in pushes:
                push_ev = pushes[ev["name"]]
                if ev["tid"] != push_ev["tid"]:
                    return {
                        "wrong_tid": ev["tid"],
                        "correct_tid": push_ev["tid"],
                        "range_name": ev["name"]
                    }
    return None
