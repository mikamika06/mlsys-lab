import sys
import numpy as np

sys.path.insert(0, ".")
from nmvalidate.validator import validate_nm_sparsity
from nmvalidate.masks import extract_nm_mask


def test_valid_sparse_matrix():
    mat = np.array([[1, 0, 0, 2], [0, 3, 4, 0]], dtype=np.float32)
    assert validate_nm_sparsity(mat, 2, 4, dim=1) is True


def test_invalid_sparse_matrix_caught():
    mat = np.array([[1, 1, 0, 2], [0, 3, 4, 0]], dtype=np.float32)
    assert validate_nm_sparsity(mat, 2, 4, dim=1) is False
