def detect_flaky(tactic_runs):
    stable = []
    for t_id, measurements in tactic_runs.items():
        mean_lat = sum(measurements) / len(measurements)
        variance = sum((m - mean_lat) ** 2 for m in measurements) / len(measurements)
        if variance < 1e-4 * (mean_lat ** 2 + 1e-5):
            stable.append(t_id)
    return sorted(stable)
