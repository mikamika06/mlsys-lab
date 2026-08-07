CONFIGS = [
    {"model": "facebook/opt-125m", "port": 8000, "tensor_parallel": 1, "max_model_len": 2048},
    {"model": "gpt2", "port": 8080, "tensor_parallel": 2, "max_model_len": 1024},
    {"model": "bigscience/bloom-560m", "port": 9000, "tensor_parallel": 1, "max_model_len": 4096},
]

def build_command(cfg):
    cmd = [
        "docker", "run", "--gpus", "all",
        "-p", f"{cfg['port']}:8000",
        "vllm/vllm-openai:latest",
        "--model", cfg["model"],
        "--tensor-parallel-size", str(cfg["tensor_parallel"]),
        "--max-model-len", str(cfg["max_model_len"])
    ]
    return cmd

def parse_health(response_status, response_body):
    if response_status == 200 and response_body.strip() in ("", "OK", "healthy"):
        return {"status": "healthy", "ready": True}
    return {"status": "unhealthy", "ready": False}

def parse_completion(payload):
    if not isinstance(payload, dict):
        raise ValueError("payload must be dict")
    choices = payload.get("choices", [])
    if not choices:
        return ""
    return choices[0].get("text", "")
