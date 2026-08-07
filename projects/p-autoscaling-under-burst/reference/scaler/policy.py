def should_admit(queue_length, max_queue, slo_headroom):
    if queue_length >= max_queue:
        return False
    if slo_headroom < 0.1:
        return False
    return True
