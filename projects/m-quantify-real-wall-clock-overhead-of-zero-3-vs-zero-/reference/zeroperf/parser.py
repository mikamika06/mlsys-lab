def parse_log(log_str):
    lines = log_str.strip().split("\n")
    times = []
    for line in lines:
        if "step_time:" in line:
            parts = line.split("step_time:")
            try:
                times.append(float(parts[1].strip()))
            except ValueError:
                pass
    return times
