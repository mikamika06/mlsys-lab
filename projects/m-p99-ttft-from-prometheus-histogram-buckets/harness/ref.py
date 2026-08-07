import math

HISTOGRAM_TEST_CASES = [
    {
        "q": 0.99,
        "buckets": [(0.05, 10), (0.1, 30), (0.25, 70), (0.5, 95), (1.0, 99), (2.0, 100), (float("inf"), 100)],
    },
    {
        "q": 0.95,
        "buckets": [(10.0, 50), (20.0, 80), (50.0, 90), (100.0, 98), (500.0, 100), (float("inf"), 100)],
    },
    {
        "q": 0.50,
        "buckets": [(1.0, 5), (5.0, 25), (10.0, 50), (20.0, 75), (float("inf"), 100)],
    },
    {
        "q": 0.90,
        "buckets": [(0.1, 0), (0.5, 0), (1.0, 100), (float("inf"), 100)],
    },
    {
        "q": 0.99,
        "buckets": [(10.0, 10), (20.0, 20), (30.0, 30), (float("inf"), 30)],
    },
]

TRIAGE_TEST_CASES = [
    (
        {"ttft_p99": 250.0, "kv_cache_usage": 1.0, "gpu_utilization": 0.98, "prefix_hit_rate": 0.85},
        50.0,
        "kv_cache_saturated",
    ),
    (
        {"ttft_p99": 300.0, "kv_cache_usage": 0.70, "gpu_utilization": 0.96, "prefix_hit_rate": 0.90},
        50.0,
        "gpu_compute_saturated",
    ),
    (
        {"ttft_p99": 280.0, "kv_cache_usage": 0.60, "gpu_utilization": 0.75, "prefix_hit_rate": 0.10},
        50.0,
        "prefix_cache_miss_spike",
    ),
    (
        {"ttft_p99": 60.0, "kv_cache_usage": 0.50, "gpu_utilization": 0.70, "prefix_hit_rate": 0.80},
        50.0,
        "nominal",
    ),
    (
        {"ttft_p99": 400.0, "kv_cache_usage": 0.80, "gpu_utilization": 0.85, "prefix_hit_rate": 0.50},
        50.0,
        "unclassified_latency_spike",
    ),
]

ALERT_TEST_CASES = [
    {
        "high": 100.0,
        "low": 50.0,
        "hold": 2,
        "inputs": [40, 110, 120, 80, 40, 30, 110, 110, 40, 40],
        "expected": [False, False, True, True, True, False, False, True, True, False],
    }
]


def calculate_histogram_quantile(q: float, buckets: list[tuple[float, float]]) -> float:
    if not buckets:
        return 0.0
    total_count = float(buckets[-1][1])
    if total_count <= 0.0:
        return 0.0
    rank = q * total_count
    if rank <= 0.0:
        return 0.0

    idx = -1
    for i, b in enumerate(buckets):
        if float(b[1]) >= rank:
            idx = i
            break

    if idx == -1:
        return float(buckets[-1][0])

    if idx == 0:
        lower_bound = 0.0
        lower_count = 0.0
        upper_bound = float(buckets[0][0])
        upper_count = float(buckets[0][1])
    else:
        lower_bound = float(buckets[idx - 1][0])
        lower_count = float(buckets[idx - 1][1])
        upper_bound = float(buckets[idx][0])
        upper_count = float(buckets[idx][1])

    if math.isinf(upper_bound):
        return lower_bound

    if upper_count == lower_count:
        return upper_bound

    fraction = (rank - lower_count) / (upper_count - lower_count)
    return lower_bound + fraction * (upper_bound - lower_bound)


def triage_system_state(metrics_snapshot: dict[str, float], baseline_ttft: float) -> str:
    if baseline_ttft <= 0.0:
        return "unknown"
    ttft = float(metrics_snapshot.get("ttft_p99", 0.0))
    kv_usage = float(metrics_snapshot.get("kv_cache_usage", 0.0))
    gpu_util = float(metrics_snapshot.get("gpu_utilization", 0.0))
    prefix_hit = float(metrics_snapshot.get("prefix_hit_rate", 1.0))

    if ttft >= 5.0 * baseline_ttft:
        if kv_usage >= 0.99:
            return "kv_cache_saturated"
        if gpu_util >= 0.95:
            return "gpu_compute_saturated"
        if prefix_hit <= 0.20:
            return "prefix_cache_miss_spike"
        return "unclassified_latency_spike"

    return "nominal"


class HysteresisAlert:
    def __init__(self, high_threshold: float, low_threshold: float, hold_periods: int = 3):
        self.high_threshold = float(high_threshold)
        self.low_threshold = float(low_threshold)
        self.hold_periods = int(hold_periods)
        self.firing = False
        self.high_count = 0
        self.low_count = 0

    def process(self, value: float) -> bool:
        val = float(value)
        if val >= self.high_threshold:
            self.high_count += 1
            self.low_count = 0
            if self.high_count >= self.hold_periods:
                self.firing = True
        elif val < self.low_threshold:
            self.low_count += 1
            self.high_count = 0
            if self.low_count >= self.hold_periods:
                self.firing = False
        else:
            self.high_count = 0
            self.low_count = 0

        return self.firing
