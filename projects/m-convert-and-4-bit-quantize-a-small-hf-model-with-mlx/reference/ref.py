CONFIGS = [
    {"model": "test-model-alpha", "quantize": True, "bits": 4, "group_size": 64},
    {"model": "test-model-beta", "quantize": True, "bits": 4, "group_size": 32},
    {"model": "test-model-gamma", "quantize": False, "bits": 16, "group_size": 0},
]

def build_convert_args(cfg):
    args = ["--model", cfg["model"]]
    if cfg["quantize"]:
        args.extend(["-q", "--q-bits", str(cfg["bits"])])
        if cfg.get("group_size", 0) > 0:
            args.extend(["--q-group-size", str(cfg["group_size"])])
    return args

def compute_speedup(runs_fp16, runs_quant):
    avg_fp16 = sum(runs_fp16) / len(runs_fp16)
    avg_quant = sum(runs_quant) / len(runs_quant)
    return round(avg_quant / avg_fp16, 4)

def validate_openai_schema(payload):
    if not isinstance(payload, dict):
        return False
    if payload.get("object") != "chat.completion":
        return False
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            return False
        message = choice.get("message")
        if not isinstance(message, dict):
            return False
        if message.get("role") not in ("assistant", "user", "system"):
            return False
        if not isinstance(message.get("content"), str):
            return False
    return True
