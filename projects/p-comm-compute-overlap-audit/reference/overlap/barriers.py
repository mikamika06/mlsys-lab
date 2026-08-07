def find_barriers(timeline):
    barriers = []
    for ev in timeline:
        if ev.get("sync", False) or ev.get("name") == "Barrier":
            barriers.append(ev.get("id", "unknown"))
    return barriers
