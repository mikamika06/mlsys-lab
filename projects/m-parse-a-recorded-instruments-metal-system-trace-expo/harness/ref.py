import random


def generate_trace_text():
    kinds = ["compute", "blit", "render", "compute", "compute"]
    lines = ["# Instruments Metal System Trace Export", "Timestamp,CommandQueue,BufferKind,DurationUs"]
    random.seed(42)
    for i in range(120):
        kind = random.choice(kinds)
        cq = f"queue_{i % 3}"
        dur = random.randint(10, 500)
        lines.append(f"100{i},{cq},{kind},{dur}")
    return "\n".join(lines)


def parse_trace(text):
    counts = {}
    for line in text.strip().splitlines():
        if line.startswith("#") or line.startswith("Timestamp"):
            continue
        parts = line.split(",")
        if len(parts) >= 3:
            kind = parts[2].strip()
            counts[kind] = counts.get(kind, 0) + 1
    return counts


def count_command_buffers(mode, op_count):
    if mode == "loop":
        return op_count
    elif mode == "graph":
        return 1
    raise ValueError("unknown mode")


def measure_overhead(cached, iterations):
    base_cost = 5.0 if cached else 50.0
    return base_cost * iterations
