def measure_backward_time(qkvpacked, q, k, v):
    if qkvpacked:
        return {"type": "packed", "time": 1.0}
    return {"type": "unpacked", "time": 1.2}
