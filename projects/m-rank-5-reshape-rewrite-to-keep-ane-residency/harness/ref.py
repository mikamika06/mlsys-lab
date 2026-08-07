import random

RANDOM = random.Random(42)

REWRITE_CASES = [
    [1, 4, 32, 8, 16, 64],
    [2, 3, 4, 5, 6, 7, 8],
    [1, 1, 16, 32, 64, 128]
]

def rewrite_shape(shape):
    if len(shape) <= 5:
        return list(shape)
    res = list(shape)
    while len(res) > 5:
        val = res.pop(0)
        res[0] *= val
    return res

PLAN_CASES = [
    {"ops": [{"target": "ANE", "cost": 10}, {"target": "CPU", "cost": 5}]},
    {"ops": [{"target": "ANE", "cost": 20}, {"target": "ANE", "cost": 10}]},
    {"ops": [{"target": "CPU", "cost": 15}, {"target": "GPU", "cost": 5}]}
]

def compute_residency_score(plan):
    total = sum(op["cost"] for op in plan["ops"])
    if total == 0:
        return 0.0
    ane = sum(op["cost"] for op in plan["ops"] if op["target"] == "ANE")
    return float(ane) / float(total)

PREDICT_CASES = [
    ({"max_rank": 4, "reshapes": 2}, {"max_rank": 6, "reshapes": 8}),
    ({"max_rank": 5, "reshapes": 1}, {"max_rank": 7, "reshapes": 5})
]

def predict_ane_friendly(arch_a, arch_b):
    score_a = -arch_a["max_rank"] * 10 - arch_a["reshapes"]
    score_b = -arch_b["max_rank"] * 10 - arch_b["reshapes"]
    if score_a >= score_b:
        return "arch_a"
    return "arch_b"
