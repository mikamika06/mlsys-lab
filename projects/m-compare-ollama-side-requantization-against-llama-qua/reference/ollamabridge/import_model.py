def verify_safetensors_dir(metadata):
    supported = ["llama", "mistral", "phi3"]
    arch = metadata.get("architecture", "")
    return arch in supported
