def translate_to_k8s_pod(spec: dict) -> dict:
    name = spec.get("name", "vllm-node")
    image = spec["image"]

    container = {
        "name": name,
        "image": image,
    }

    if spec.get("command"):
        container["command"] = spec["command"]

    env_vars = []
    for k in sorted(spec.get("env", {}).keys()):
        env_vars.append({"name": k, "value": str(spec["env"][k])})
    if env_vars:
        container["env"] = env_vars

    ports = []
    for p in sorted(spec.get("ports", []), key=lambda x: x.get("container", 0)):
        ports.append({"containerPort": p["container"]})
    if ports:
        container["ports"] = ports

    volume_mounts = []
    volumes = []

    for idx, vol in enumerate(sorted(spec.get("volumes", []), key=lambda x: x["container"])):
        v_name = f"vol-{idx}"
        mount = {"name": v_name, "mountPath": vol["container"]}
        if vol.get("mode") == "ro":
            mount["readOnly"] = True
        volume_mounts.append(mount)
        volumes.append({
            "name": v_name,
            "hostPath": {"path": vol["host"]}
        })

    shm_size = spec.get("shm_size", "64m")
    volume_mounts.append({
        "name": "dshm",
        "mountPath": "/dev/shm"
    })
    volumes.append({
        "name": "dshm",
        "emptyDir": {
            "medium": "Memory",
            "sizeLimit": shm_size
        }
    })

    container["volumeMounts"] = volume_mounts

    gpus = spec.get("gpus")
    if gpus is not None:
        container["resources"] = {
            "limits": {
                "nvidia.com/gpu": str(gpus)
            }
        }

    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": name
        },
        "spec": {
            "containers": [container],
            "volumes": volumes
        }
    }
