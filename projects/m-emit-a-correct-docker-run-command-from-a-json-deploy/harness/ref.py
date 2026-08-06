SPECS = [
    {
        "name": "vllm-single",
        "image": "vllm/vllm-openai:latest",
        "gpus": 1,
        "shm_size": "2g",
        "ports": {"8000": 8000},
        "env": {"MODEL": "meta-llama/Llama-3-8B-Instruct"},
        "volumes": {"/data": "/data"},
        "command_args": ["--model", "meta-llama/Llama-3-8B-Instruct"]
    },
    {
        "name": "vllm-multi",
        "image": "vllm/vllm-openai:latest",
        "gpus": 4,
        "shm_size": "16g",
        "ports": {"8000": 8000},
        "env": {"MODEL": "meta-llama/Llama-3-70B-Instruct"},
        "volumes": {},
        "ipc": "host",
        "command_args": ["--model", "meta-llama/Llama-3-70B-Instruct", "--tensor-parallel-size", "4"]
    },
    {
        "name": "vllm-custom",
        "image": "vllm/vllm-openai:latest",
        "gpus": "all",
        "ports": {"8000": 8000},
        "env": {},
        "volumes": {"/tmp/cache": "/root/.cache/huggingface"},
        "command_args": ["--model", "mistralai/Mistral-7B-v0.1"]
    }
]

LOGS = [
    "RuntimeError: NCCL error: unhandled system error in torch.distributed",
    "CUDA out of memory during allocation of 4.20GB",
    "Container started successfully without issues."
]

import shlex


def emit_docker_run(spec: dict) -> str:
    cmd = ["docker", "run", "--rm", "-it"]

    if spec.get("ipc") == "host":
        cmd.extend(["--ipc=host"])
    else:
        shm_size = spec.get("shm_size")
        if shm_size:
            cmd.extend([f"--shm-size={shm_size}"])

    gpus = spec.get("gpus")
    if gpus is not None:
        if isinstance(gpus, int):
            cmd.extend(["--gpus", str(gpus)])
        elif isinstance(gpus, str):
            cmd.extend(["--gpus", gpus])

    ports = spec.get("ports", {})
    for host_port, container_port in ports.items():
        cmd.extend(["-p", f"{host_port}:{container_port}"])

    env = spec.get("env", {})
    for k, v in env.items():
        cmd.extend(["-e", f"{k}={v}"])

    volumes = spec.get("volumes", {})
    for host_path, container_path in volumes.items():
        cmd.extend(["-v", f"{host_path}:{container_path}"])

    image = spec.get("image")
    if image:
        cmd.append(image)

    command_args = spec.get("command_args", [])
    if command_args:
        cmd.extend(command_args)

    return " ".join(shlex.quote(c) for c in cmd)


def diagnose_failure(logs: str, spec: dict) -> dict:
    issues = []
    shm_size = spec.get("shm_size")
    ipc = spec.get("ipc")
    gpus = spec.get("gpus", 0)

    if isinstance(gpus, int) and gpus > 1 and ipc != "host":
        if not shm_size:
            issues.append("shm_too_small")
        else:
            pass

    if "RuntimeError: NCCL error" in logs or "torch.distributed" in logs or "Bus error" in logs:
        if not shm_size and ipc != "host":
            issues.append("shm_too_small")

    if "CUDA out of memory" in logs:
        issues.append("cuda_oom")

    if not issues and ("segmentation fault" in logs.lower() or "sigsegv" in logs.lower()):
        issues.append("shm_too_small")

    return {
        "issues": list(set(issues)),
        "is_multi_gpu": bool(isinstance(gpus, int) and gpus > 1 or isinstance(gpus, str) and gpus != "0"),
        "ipc_mode": ipc,
        "shm_size": shm_size
    }
