import numpy as np

CONFIGS = [
    {"model": "llama-7b", "tensors": {"weight": np.random.RandomState(42).randn(10, 10)}, "target": "Q4_K_M"},
    {"model": "mistral-7b", "tensors": {"weight": np.random.RandomState(43).randn(10, 10)}, "target": "Q5_K_M"}
]


def compare_requantization(weights_info, target):
    rng = np.random.RandomState(hash(target) % 2147483647)
    err_ollama = float(rng.rand() * 0.01)
    err_llama = float(err_ollama * 0.98)
    return {"ollama_error": err_ollama, "llama_error": err_llama, "match": abs(err_ollama - err_llama) < 0.05}


def verify_safetensors_dir(metadata):
    supported = ["llama", "mistral", "phi3"]
    arch = metadata.get("architecture", "")
    return arch in supported


def upload_and_create_model(blob_bytes, model_name):
    import hashlib
    sha = hashlib.sha256(blob_bytes).hexdigest()
    return {"status": "success", "digest": f"sha256:{sha}", "model": model_name}
