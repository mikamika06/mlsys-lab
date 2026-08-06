def docker_to_k8s_pod(spec: dict) -> dict:
    name = spec.get("name", "vllm-node")
    image = spec.get("image", "vllm/vllm-openai:latest")
    env_list = [{"name": str(k), "value": str(v)} for k, v in sorted((spec.get("env") or {}).items())]
    container = {
        "name": name,
        "image": image,
        "env": env_list,
        "volumeMounts": []
    }
    if spec.get("entrypoint"):
        container["command"] = [str(spec["entrypoint"])]
    if spec.get("args"):
        container["args"] = [str(a) for a in spec["args"]]
    volumes = []
    shm_needed = spec.get("ipc") == "host" or "shm_size" in spec or spec.get("tensor_parallel_size", 1) > 1
    if shm_needed:
        container["volumeMounts"].append({
            "name": "dshm",
            "mountPath": "/dev/shm"
        })
        vol = {"name": "dshm", "emptyDir": {"medium": "Memory"}}
        if "shm_size" in spec:
            vol["emptyDir"]["sizeLimit"] = str(spec["shm_size"])
        volumes.append(vol)
    for host_v, container_v in sorted((spec.get("volumes") or {}).items(), key=lambda x: str(x[0])):
        v_name = f"vol-{len(volumes)}"
        container["volumeMounts"].append({"name": v_name, "mountPath": container_v})
        volumes.append({"name": v_name, "hostPath": {"path": host_v}})
    pod = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": name},
        "spec": {
            "containers": [container],
            "volumes": volumes
        }
    }
    return pod
