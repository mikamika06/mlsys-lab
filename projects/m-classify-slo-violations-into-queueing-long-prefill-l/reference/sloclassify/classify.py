def classify_violation(req, slo_target):
    if req["total_latency"] <= slo_target:
        return "none"
    qt = req["queue_time"]
    pt = req["prefill_time"]
    ot = req["output_time"]
    if qt >= pt and qt >= ot:
        return "queueing"
    elif pt >= ot:
        return "long-prefill"
    else:
        return "long-output"
