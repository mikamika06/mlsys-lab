def parse_utilization(log_lines):
    utils = []
    for line in log_lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        utils.append(float(parts[1]) / float(parts[2]))
    if not utils:
        return {"mean": 0.0, "max": 0.0}
    return {"mean": sum(utils) / len(utils), "max": max(utils)}
