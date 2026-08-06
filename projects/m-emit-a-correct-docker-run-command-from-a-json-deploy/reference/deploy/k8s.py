def docker_to_k8s(spec: dict) -> dict:
    pod = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": spec.get("name", "vllm-serving-pod")
        },
        "spec": {
            "containers": [
                {
                    "name": spec.get("name", "vllm-container"),
                    "image": spec.get("image", "vllm/vllm-openai:latest"),
                    "env": [{"name": k, "value": str(v)} for k, v in spec.get("env", {}).items()],
                    "ports": [{"containerPort": int(v)} for k, v in spec.get("ports", {}).items()],
                    "volumeMounts": [
                        {"mountPath": container_path, "name": f"vol-{i}"}
                        for i, (host_path, container_path) in enumerate(spec.get("volumes", {}).items())
                    ]
                }
            ],
            "volumes": [
                {"name": f"vol-{i}", "hostPath": {"path": host_path}}
                for i, (host_path, container_path) in enumerate(spec.get("volumes", {}).items())
            ]
        }
    }

    gpus = spec.get("gpus")
    if gpus:
        count = str(gpus) if not isinstance(gpus, int) else str(gpus)
        pod["spec"]["containers"][0]["resources"] = {
            "limits": {"nvidia.com/gpu": count},
            "requests": {"nvidia.com/gpu": count}
        }

    shm_size = spec.get("shm_size")
    ipc = spec.get("ipc")
    if ipc == "host" or shm_size:
        size_limit = shm_size if shm_size else "2Gi"
        pod["spec"]["volumes"].append({
            "name": "dshm",
            "emptyDir": {
                "medium": "Memory",
                "sizeLimit": size_limit
            }
        })
        pod["spec"]["containers"][0]["volumeMounts"].append({
            "mountPath": "/dev/shm",
            "name": "dshm"
        })

    return pod
