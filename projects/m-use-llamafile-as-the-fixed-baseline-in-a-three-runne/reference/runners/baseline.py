def setup_baseline(config):
    return {
        "name": "llamafile",
        "mode": "fixed_baseline",
        "parameters": config.get("parameters", 7),
        "quantization": config.get("quantization", "Q4_0"),
        "ready": True
    }
