import random

def get_test_inputs():
    rnd = random.Random(42)
    x = [rnd.uniform(-10.0, 10.0) for _ in range(64)]
    y = [rnd.uniform(-10.0, 10.0) for _ in range(64)]
    return x, y, 16

def get_test_tree():
    return {
        "name": "root",
        "duration": 200.0,
        "children": [
            {"name": "load", "duration": 50.0, "children": []},
            {"name": "compute", "duration": 120.0, "children": []},
            {"name": "store", "duration": 30.0, "children": []}
        ]
    }

def get_test_sweeps():
    return [
        {"config": {"BLOCK_SIZE": 16, "NUM_WARPS": 2}, "latency": 45.5},
        {"config": {"BLOCK_SIZE": 32, "NUM_WARPS": 4}, "latency": 12.1},
        {"config": {"BLOCK_SIZE": 64, "NUM_WARPS": 4}, "latency": 18.3},
        {"config": {"BLOCK_SIZE": 128, "NUM_WARPS": 8}, "latency": 25.0}
    ]
