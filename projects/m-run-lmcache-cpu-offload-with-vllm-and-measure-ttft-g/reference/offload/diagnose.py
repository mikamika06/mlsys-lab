def diagnose_transfer_log(log_lines):
    results = []
    for line in log_lines:
        parts = line.strip().split(",")
        if len(parts) < 3:
            results.append("unknown")
            continue
        duration = float(parts[1])
        bytes_transferred = float(parts[2])
        throughput = bytes_transferred / (duration + 1e-6)
        if throughput < 10.0:
            results.append("pcie_bottleneck")
        elif duration > 50.0:
            results.append("lock_contention")
        else:
            results.append("healthy")
    return results
