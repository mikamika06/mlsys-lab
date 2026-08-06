import sys
import numpy as np

sys.path.insert(0, ".")
from palettize.kmeans import kmeans_palettize
from palettize.bytes import exact_tensor_bytes
from palettize.compare import compare_scalar_vector


def test_kmeans_shapes_and_types():
    t = np.random.default_rng(0).normal(size=(16, 16)).astype(np.float32)
    c, l = kmeans_palettize(t, 3, vector_length=1)
    assert c.shape == (8, 1)
    assert l.ndim == 1


def test_exact_bytes_calculation():
    b = exact_tensor_bytes(100, 4, vector_length=1, centroid_dtype_bytes=4)
    assert b > 0


def test_compare_scalar_vector_keys():
    t = np.random.default_rng(0).normal(size=(8, 8)).astype(np.float32)
    res = compare_scalar_vector(t, 2)
    assert "scalar_mse" in res
    assert "vector_mse" in res
