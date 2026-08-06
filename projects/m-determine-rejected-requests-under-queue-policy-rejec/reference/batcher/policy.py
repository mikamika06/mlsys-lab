def compute_rejected_requests(queue_size: int, max_queue_size: int, incoming_count: int, policy: str) -> int:
    if policy != "REJECT":
        return 0
    available_space = max(0, max_queue_size - queue_size)
    if incoming_count > available_space:
        return incoming_count - available_space
    return 0
