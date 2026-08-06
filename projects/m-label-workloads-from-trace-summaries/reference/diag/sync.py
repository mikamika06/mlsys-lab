def find_hidden_syncs(event_log):
    return [i for i, ev in enumerate(event_log) if ev.get("is_blocking", False)]
