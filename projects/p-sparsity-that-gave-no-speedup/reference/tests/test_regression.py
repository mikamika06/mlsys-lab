import numpy as np
import sparsity.checkpoint as chk

def test_checkpoint_size_smaller():
    m = np.array([[1, 0, 0, 2], [0, 3, 4, 0]])
    size = chk.checkpoint_size(m)
    assert size == 8 * 1.125

def test_checkpoint_size_dense():
    m = np.array([[1, 1, 1, 1], [0, 0, 0, 0]])
    size = chk.checkpoint_size(m)
    assert size == 8 * 2.0
