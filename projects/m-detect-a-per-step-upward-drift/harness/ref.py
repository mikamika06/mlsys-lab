import random

def get_test_cases():
    random.seed(42)
    series_up = [100 + i * 2 + random.uniform(-0.1, 0.1) for i in range(10)]
    series_flat = [100 + random.uniform(-0.1, 0.1) for i in range(10)]
    return [series_up, series_flat]

def get_snapshot_cases():
    return [
        [{"allocated": 1500, "peak": 2500, "active": 200},
         {"allocated": 100, "peak": 150, "active": 50},
         {"allocated": 200, "peak": 300, "active": 600},
         {"allocated": 1600, "peak": 2600, "active": 100},
         {"allocated": 50, "peak": 80, "active": 10},
         {"allocated": 100, "peak": 120, "active": 700}]
    ]
