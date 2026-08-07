SPECS = [
    {"peak_flops": 312e12, "peak_bandwidth": 1.55e12, "expected_ridge": 201.29032258064518},
    {"peak_flops": 100e12, "peak_bandwidth": 500e9, "expected_ridge": 200.0},
    {"peak_flops": 50e12, "peak_bandwidth": 250e9, "expected_ridge": 200.0},
]

CONFIGS = [
    {"intensity": 10.0, "ridge": 200.0, "expected": "memory-bound"},
    {"intensity": 250.0, "ridge": 200.0, "expected": "compute-bound"},
    {"intensity": 50.0, "ridge": 100.0, "expected": "memory-bound"},
    {"intensity": 150.0, "ridge": 100.0, "expected": "compute-bound"},
]

def compute_ridge_point(peak_flops: float, peak_bandwidth: float) -> float:
    return peak_flops / peak_bandwidth

def classify_decode(arithmetic_intensity: float, ridge_point: float) -> str:
    if arithmetic_intensity < ridge_point:
        return "memory-bound"
    return "compute-bound"

def measure_crossover_ratio(lat_batch1: float, lat_batch32: float) -> float:
    return lat_batch32 / max(lat_batch1, 1e-9)
