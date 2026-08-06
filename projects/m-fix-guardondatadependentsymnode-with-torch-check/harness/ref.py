CONFIGS = [
    {"input_dim": 16, "max_limit": 1024, "mode": "strict"},
    {"input_dim": 32, "max_limit": 2048, "mode": "flexible"},
    {"input_dim": 64, "max_limit": 4096, "mode": "strict"}
]

TRACES = [
    {"trace": [10, 20, 30], "dynamic": True},
    {"trace": [128, 128, 128], "dynamic": False},
    {"trace": [16, 32, 64], "dynamic": True}
]

def fix_code(cfg):
    return f"torch._check(x.shape[0] <= {cfg['max_limit']}); return x * {cfg['input_dim']}"

def predict_dynamic(trace):
    values = trace["trace"]
    return len(set(values)) > 1 or trace["dynamic"]
