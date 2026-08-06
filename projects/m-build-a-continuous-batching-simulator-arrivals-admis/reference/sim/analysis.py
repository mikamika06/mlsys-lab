def occupancy_histogram(log_entries, max_bins):
    counts = [0] * (max_bins + 1)
    for entry in log_entries:
        cnt = entry["active_count"]
        if cnt <= max_bins:
            counts[cnt] += 1
    return counts
