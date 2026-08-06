REVISIONS = [
    {"rev1": {"blob_a": 100, "blob_b": 200}, "rev2": {"blob_b": 200, "blob_c": 300}},
    {"rev1": {"x": 10, "y": 20, "z": 30}, "rev2": {"y": 20, "z": 30, "w": 40}},
    {"rev1": {"m": 500}, "rev2": {"m": 500, "n": 600}}
]

BUDGET_CASES = [
    {"pull_size": 1000, "weight_size": 5000, "compile_factor": 2.0, "pull_speed": 100, "weight_speed": 500},
    {"pull_size": 2000, "weight_size": 10000, "compile_factor": 1.5, "pull_speed": 200, "weight_speed": 1000}
]

def audit_reused_blobs(rev1, rev2):
    set1 = set(rev1.items())
    set2 = set(rev2.items())
    reused = dict(set1.intersection(set2))
    only_rev1 = dict(set1 - set2)
    only_rev2 = dict(set2 - set1)
    return {"reused": reused, "only_rev1": only_rev1, "only_rev2": only_rev2}

def predict_ready_time(pull_size, weight_size, compile_factor, pull_speed, weight_speed):
    t_pull = pull_size / pull_speed
    t_weight = weight_size / weight_speed
    t_compile = (pull_size + weight_size) * compile_factor / 1000.0
    return float(t_pull + t_weight + t_compile)
