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
