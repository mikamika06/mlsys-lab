import random

SCHEDULE_TYPES = ["gpipe", "1f1b", "zero_bubble", "interleaved"]

def generate_mock_actions(schedule_type, seed=42):
    actions = []
    if schedule_type == "gpipe":
        for mb in range(4):
            actions.append({"type": "FORWARD", "microbatch": mb})
        for mb in range(4):
            actions.append({"type": "BACKWARD", "microbatch": mb})
    elif schedule_type == "1f1b":
        for mb in range(2):
            actions.append({"type": "FORWARD", "microbatch": mb})
        for mb in range(2):
            actions.append({"type": "BACKWARD", "microbatch": mb})
            actions.append({"type": "FORWARD", "microbatch": mb + 2})
        for mb in range(2, 4):
            actions.append({"type": "BACKWARD", "microbatch": mb})
    elif schedule_type == "zero_bubble":
        for mb in range(3):
            actions.append({"type": "FORWARD", "microbatch": mb})
        for mb in range(2):
            actions.append({"type": "BACKWARD_WEIGHT", "microbatch": mb})
            actions.append({"type": "BACKWARD", "microbatch": mb})
        for mb in range(2, 3):
            actions.append({"type": "BACKWARD", "microbatch": mb})
    else:
        for v in [0, 1]:
            actions.append({"type": "VIRTUAL_FORWARD", "virtual_stage": v, "microbatch": 0})
        for v in [0, 1]:
            actions.append({"type": "VIRTUAL_BACKWARD", "virtual_stage": v, "microbatch": 0})
    return actions

def identify_schedule(actions):
    types = [a["type"] for a in actions]
    if any("WEIGHT" in t for t in types):
        return "zero_bubble"
    if any("VIRTUAL" in t for t in types):
        return "interleaved"
    fw_indices = [i for i, t in enumerate(types) if t == "FORWARD"]
    bw_indices = [i for i, t in enumerate(types) if t == "BACKWARD"]
    if fw_indices and bw_indices and max(fw_indices) < min(bw_indices):
        return "gpipe"
    return "1f1b"

def generate_mock_logs(bubble_ratio, seed=42):
    compute_time = 1000.0
    bubble_time = compute_time * bubble_ratio
    log_lines = [
        f"RANK 0: compute_time={compute_time:.2f}ms",
        f"RANK 0: idle_bubble_time={bubble_time:.2f}ms",
        f"RANK 1: compute_time={compute_time:.2f}ms",
        f"RANK 1: idle_bubble_time={bubble_time:.2f}ms"
    ]
    return log_lines

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
