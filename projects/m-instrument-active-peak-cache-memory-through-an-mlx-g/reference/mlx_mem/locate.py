def locate_ceiling(logs, ceiling):
    for line in logs:
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            step = int(parts[0])
            mem = int(parts[1])
            if mem > ceiling:
                return step
    return -1
