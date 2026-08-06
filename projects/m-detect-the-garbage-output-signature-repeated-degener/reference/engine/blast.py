def estimate_blast(batch_requests, crash_index):
    lost = []
    retried = 0
    for req in batch_requests:
        if req["index"] >= crash_index:
            lost.append(req["id"])
            if req.get("retry_count", 0) < req.get("max_retries", 3):
                retried += 1
    return {
        "lost_count": len(lost),
        "lost_ids": lost,
        "retried_count": retried
    }
