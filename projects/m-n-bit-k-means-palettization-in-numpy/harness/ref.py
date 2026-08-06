import numpy as np
from reference.palettize.kmeans import kmeans_palettize
from reference.palettize.bytes import exact_tensor_bytes
from reference.palettize.compare import compare_scalar_vector

TENSORS = [
    np.random.default_rng(42).normal(size=(16, 16)).astype(np.float32),
    np.random.default_rng(123).uniform(-1, 1, size=(32, 16)).astype(np.float32),
    np.random.default_rng(7).standard_normal(size=(8, 32)).astype(np.float32)
]
