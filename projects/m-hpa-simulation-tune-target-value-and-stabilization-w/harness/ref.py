import random

CONFIGS = [
    {"load": [50, 120, 250, 300, 150, 80], "target": 100, "window": 2, "min": 1, "max": 5},
    {"load": [10, 40, 90, 200, 180, 50], "target": 80, "window": 3, "min": 1, "max": 8},
    {"load": [100, 100, 100, 100, 100], "target": 50, "window": 1, "min": 2, "max": 4},
]

TIMELINES = [
    [{"step": 1, "oscillation_detected": True, "responsible_parameter": "stabilization_window"}]
]

SESSIONS = [
    [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]
]
