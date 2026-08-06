def reorder_logs(logs):
    parsed = []
    for line in logs:
        parts = line.split("|")
        if len(parts) >= 3:
            try:
                ts = float(parts[0].strip())
                pid = int(parts[1].strip())
                seq = int(parts[2].strip())
                parsed.append((ts, pid, seq, line))
            except ValueError:
                parsed.append((0.0, 0, 0, line))
        else:
            parsed.append((0.0, 0, 0, line))
    parsed.sort(key=lambda x: (x[0], x[1], x[2]))
    return [item[3] for item in parsed]
