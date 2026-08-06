def generate_dockerfile(config):
    cuda_ver = config.get("cuda_version", "12.1")
    torch_ver = config.get("torch_version", "2.3.0")
    return f"""FROM nvidia/cuda:{cuda_ver}-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3-pip git
RUN pip install --no-cache-dir torch=={torch_ver}
RUN pip install --no-cache-dir flash-attn==2.5.8 --no-build-isolation
"""
