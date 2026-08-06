def build_docker_run(spec: dict) -> str:
    parts = ["docker", "run", "-d"]
    if spec.get("name"):
        parts.extend(["--name", str(spec["name"])])
    if spec.get("gpus"):
        gpus = str(spec["gpus"])
        parts.extend(["--gpus", f'"{gpus}"' if " " in gpus else gpus])
    if spec.get("ipc"):
        parts.extend(["--ipc", str(spec["ipc"])])
    if spec.get("shm_size"):
        parts.extend(["--shm-size", str(spec["shm_size"])])
    for host_p, container_p in sorted((spec.get("ports") or {}).items(), key=lambda x: str(x[0])):
        parts.extend(["-p", f"{host_p}:{container_p}"])
    for k, v in sorted((spec.get("env") or {}).items()):
        parts.extend(["-e", f"{k}={v}"])
    for host_v, container_v in sorted((spec.get("volumes") or {}).items(), key=lambda x: str(x[0])):
        parts.extend(["-v", f"{host_v}:{container_v}"])
    if spec.get("entrypoint"):
        parts.extend(["--entrypoint", str(spec["entrypoint"])])
    parts.append(str(spec.get("image", "ubuntu:latest")))
    if spec.get("args"):
        parts.extend([str(a) for a in spec["args"]])
    return " ".join(parts)
