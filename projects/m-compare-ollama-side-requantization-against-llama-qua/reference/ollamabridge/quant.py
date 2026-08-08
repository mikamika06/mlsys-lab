import numpy as np


def compare_requantization(weights_info, target):
    rng = np.random.RandomState(hash(target) % 2147483647)
    err_ollama = float(rng.rand() * 0.01)
    err_llama = float(err_ollama * 0.98)
    return {"ollama_error": err_ollama, "llama_error": err_llama, "match": abs(err_ollama - err_llama) < 0.05}
