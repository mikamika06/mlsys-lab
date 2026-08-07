import numpy as np


def generate_24_sparse_matrix(M: int, K: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    mat = np.zeros((M, K), dtype=np.float32)
    num_blocks = K // 4
    for r in range(M):
        for c in range(num_blocks):
            cols = rng.choice(4, size=2, replace=False)
            mat[r, c * 4 + cols[0]] = float(rng.randn() + 1.0)
            mat[r, c * 4 + cols[1]] = float(rng.randn() + 1.0)
    return mat


def generate_dense_matrix(M: int, K: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return (rng.randn(M, K) + 0.5).astype(np.float32)


def test_shapes() -> list:
    return [
        {"shape": (32, 128, 128), "is_24": False, "expected_path": "dense_unsupported_pattern"},
        {"shape": (32, 128, 100), "is_24": True, "expected_path": "dense_fallback_misaligned"},
        {"shape": (4, 128, 128), "is_24": True, "expected_path": "dense_fallback_small_batch"},
        {"shape": (32, 128, 128), "is_24": True, "expected_path": "sparse_24_tensor_core"},
    ]


def hw_config() -> dict:
    return {"peak_tflops": 312.0, "bandwidth_gbps": 2000.0, "dtype_bytes": 2}


def workloads() -> list:
    return [
        {"shape": (4, 4096, 4096), "is_24": True, "expected_speedup": False},
        {"shape": (128, 4096, 4096), "is_24": True, "expected_speedup": True},
    ]
