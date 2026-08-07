TOOLS = [
    ("perf_cpu_sampling", "sampling"),
    ("nvtx_push_pop", "instrumentation"),
    ("rr_deterministic_replay", "replay"),
    ("nsys_periodic_backtrace", "sampling"),
    ("kineto_activity_api", "instrumentation"),
    ("rr_execution_log", "replay")
]

MISS_BOUND_CASES = [
    {"interval": 10.0, "duration": 2.0, "total": 1000.0},
    {"interval": 5.0, "duration": 5.0, "total": 500.0},
    {"interval": 20.0, "duration": 1.0, "total": 2000.0}
]

PROFILERS = ["none", "sampling_low", "instrumentation_full", "sampling_high", "replay_trace"]

WORKLOAD_METRICS = {
    "none": 100.0,
    "sampling_low": 102.0,
    "instrumentation_full": 145.0,
    "sampling_high": 108.0,
    "replay_trace": 180.0
}


def classify_mechanisms(tools):
    return {name: cat for name, cat in tools}


def calculate_miss_bound(interval, duration, total):
    if duration >= interval:
        return 0.0
    return (interval - duration) / interval


def rank_profilers(profilers, metrics):
    return sorted(profilers, key=lambda p: metrics[p])
