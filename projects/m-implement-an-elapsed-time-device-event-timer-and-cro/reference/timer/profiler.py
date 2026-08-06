def verify_sync(profile):
    if not profile["synced"] and profile["host_ms"] > profile["kernel_ms"]:
        raise ValueError("Unsynced timing contaminated by host overhead")
    return True
