BUILD_TIMES_NC = [120.5, 240.0, 95.0]
BUILD_TIMES_WC = [12.0, 24.5, 10.0]
LEVELS = [1, 2, 3, 4]
LATS = [25.0, 18.0, 17.5, 17.5]
COMPS = [5.0, 15.0, 60.0, 250.0]
TACTIC_RUNS = {"tact_a": [10.1, 10.2, 10.1], "tact_b": [10.0, 25.0, 10.0]}

def compute_payoff(nc, wc):
    return [n / c for n, c in zip(nc, wc)]

def find_knee(lvl, lats, comps):
    scores = [(l, lat * ct) for l, lat, ct in zip(lvl, lats, comps)]
    return min(scores, key=lambda x: x[1])[0]

def detect_flaky(runs):
    stable = []
    for t_id, measurements in runs.items():
        mean_lat = sum(measurements) / len(measurements)
        variance = sum((m - mean_lat) ** 2 for m in measurements) / len(measurements)
        if variance < 1e-3 * (mean_lat ** 2 + 1e-5):
            stable.append(t_id)
    return sorted(stable)
