def should_prefetch(session_id, request_queue, cpu_tier):
    if session_id not in cpu_tier:
        return False
    for req in request_queue:
        if req.get("session_id") == session_id:
            return True
    return False
