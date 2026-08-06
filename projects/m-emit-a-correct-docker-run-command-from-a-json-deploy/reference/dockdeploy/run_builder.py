def build_docker_run(spec: dict) -> str:
    parts = ["docker", "run", "-d"]

    if spec.get("name"):
        parts.extend(["--name", spec["name"]])

    if spec.get("ipc"):
        parts.extend(["--ipc", spec["ipc"]])

    if spec.get("shm_size"):
        parts.extend(["--shm-size", spec["shm_size"]])

    gpus = spec.get("gpus")
    if gpus is not None:
        parts.extend(["--gpus", str(gpus)])

    for port in sorted(spec.get("ports", []), key=lambda x: (x.get("host"), x.get("container"))):
        parts.extend(["-p", f"{port['host']}:{port['container']}"])

    for vol in sorted(spec.get("volumes", []), key=lambda x: (x.get("host"), x.get("container"))):
        mode = f":{vol['mode']}" if "mode" in vol else ""
        parts.extend(["-v", f"{vol['host']}:{vol['container']}{mode}"])

    env = spec.get("env", {})
    for k in sorted(env.keys()):
        parts.extend(["-e", f"{k}={env[k]}"])

    parts.append(spec["image"])

    if spec.get("command"):
        parts.extend(spec["command"])

    return " ".join(parts)
