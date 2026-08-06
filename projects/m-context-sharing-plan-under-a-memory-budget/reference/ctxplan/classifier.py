def classify_oom(event_log, workspace_limit):
    phase = event_log.get("phase", "")
    peak = event_log.get("peak_memory", 0)
    if phase == "build" or peak > workspace_limit:
        return "build"
    return "runtime"
