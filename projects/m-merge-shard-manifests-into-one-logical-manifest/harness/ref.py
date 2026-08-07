import random

def get_test_cases():
    rng = random.Random(42)
    filenames_good = [
        "llama-0001-of-0003.gguf",
        "llama-0002-of-0003.gguf",
        "llama-0003-of-0003.gguf",
    ]
    filenames_bad = [
        "llama-0000-of-0003.gguf",
        "llama-0001-of-0000.gguf",
        "llama-1-of-3.gguf",
        "random.gguf",
    ]
    manifests = [
        {"version": 1, "size": 1024, "tensors": {"weight_a": {"offset": 0, "shape": [10]}}},
        {"version": 1, "size": 2048, "tensors": {"weight_b": {"offset": 0, "shape": [20]}}},
    ]
    return {
        "filenames_good": filenames_good,
        "filenames_bad": filenames_bad,
        "manifests": manifests,
    }
