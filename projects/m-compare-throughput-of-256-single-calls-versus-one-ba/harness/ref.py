import numpy as np


def get_test_inputs():
    rng = np.random.default_rng(1337)
    return [str(rng.integers(0, 100000)) for _ in range(256)]


def mock_single_endpoint(text):
    rng = np.random.default_rng(abs(hash(text)) % (2**31))
    arr = rng.normal(size=(32,))
    arr /= np.linalg.norm(arr)
    return arr


def mock_batched_endpoint(texts):
    results = []
    for t in texts:
        rng = np.random.default_rng(abs(hash(t)) % (2**31))
        arr = rng.normal(size=(32,))
        arr /= np.linalg.norm(arr)
        results.append(arr)
    return np.array(results)


def get_reference_embeddings():
    rng = np.random.default_rng(42)
    embs = rng.normal(size=(10, 32))
    embs /= np.linalg.norm(embs, axis=-1, keepdims=True)
    return embs
