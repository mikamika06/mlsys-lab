def detect_silent_fallback(events):
    fallbacks = []
    for ev in events:
        if getattr(ev, "device", "mps") != "mps":
            fallbacks.append(getattr(ev, "name", "unknown"))
    return fallbacks
