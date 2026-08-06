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
