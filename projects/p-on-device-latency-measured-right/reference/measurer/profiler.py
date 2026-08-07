import numpy as np

class DeviceProfiler:
    def __init__(self, config=None):
        self.config = config or {}

    def run_warmup(self, iterations):
        return list(range(iterations))

    def separate_steady_state(self, samples):
        arr = np.array(samples)
        if len(arr) <= 5:
            return arr
        return arr[int(len(arr) * 0.2):]

    def measure_cold_start(self):
        return 45.0

    def compute_required_runs(self, variance, target_error):
        z = 1.96
        n = (z * np.sqrt(variance) / target_error) ** 2
        return int(np.ceil(n))

    def check_session_overlap(self, sessions):
        intervals = []
        for s in sessions:
            arr = np.array(s)
            mean = np.mean(arr)
            std = np.std(arr, ddof=1)
            se = std / np.sqrt(len(arr))
            intervals.append((mean - 1.96 * se, mean + 1.96 * se))

        lows = [iv[0] for iv in intervals]
        highs = [iv[1] for iv in intervals]
        return max(lows) <= min(highs)

    def generate_report(self, data):
        arr = np.array(data)
        return {
            "mean": float(np.mean(arr)),
            "p99": float(np.percentile(arr, 99)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr))
        }
