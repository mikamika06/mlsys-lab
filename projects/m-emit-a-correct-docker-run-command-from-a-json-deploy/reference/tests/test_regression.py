import sys
sys.path.insert(0, ".")
from deploy.k8s import docker_to_k8s


def test_docker_to_k8s_shm_mount():
    spec = {
        "name": "test-vllm",
        "image": "vllm/vllm-openai:latest",
        "gpus": 4,
        "shm_size": "8g"
    }
    pod = docker_to_k8s(spec)
    volumes = pod["spec"]["volumes"]
    shm_vols = [v for v in volumes if v.get("name") == "dshm"]
    assert len(shm_vols) == 1, "Missing dshm volume in Kubernetes Pod spec"
    assert "emptyDir" in shm_vols[0], "dshm volume must be backed by emptyDir"
    assert shm_vols[0]["emptyDir"].get("medium") == "Memory", "dshm emptyDir must use Memory medium"


def test_docker_to_k8s_gpu_resources():
    spec = {
        "name": "test-gpu",
        "image": "vllm/vllm-openai:latest",
        "gpus": 2
    }
    pod = docker_to_k8s(spec)
    limits = pod["spec"]["containers"][0]["resources"]["limits"]
    assert limits.get("nvidia.com/gpu") == "2", "GPU limits incorrectly translated"
