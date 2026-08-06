def compute_bubble_fraction(log_lines):
    total_compute = 0.0
    total_bubble = 0.0
    for line in log_lines:
        if "compute_time=" in line:
            total_compute += float(line.split("compute_time=")[1].split("ms")[0])
        elif "idle_bubble_time=" in line:
            total_bubble += float(line.split("idle_bubble_time=")[1].split("ms")[0])
    if total_compute + total_bubble == 0.0:
        return 0.0
    return total_bubble / (total_compute + total_bubble)
