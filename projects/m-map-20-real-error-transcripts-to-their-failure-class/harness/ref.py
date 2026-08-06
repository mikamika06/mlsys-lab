ERROR_ITEMS = [
    {
        "id": i,
        "transcript": f"[ERROR] code {1000 + i}: failure in subsystem {i}",
        "failure_class": f"CLASS_{i % 5}",
        "root_cause": f"cause_{i}",
        "fix": f"fix_{i}",
        "model_size_b": 7 + i,
        "ctx": 2048,
        "tp": 2,
        "max_vram": 80.0
    }
    for i in range(20)
]

def get_error_items():
    return ERROR_ITEMS

def predict_fit(model_size_b, ctx, tp, vram_gb):
    return (model_size_b * 2.0 + (ctx / 2048.0) * (8.0 / tp)) <= vram_gb
