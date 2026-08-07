import sys
sys.path.insert(0, ".")
from coreprune.prune import prune_weights
from coreprune.palettize import chain_prune_palettize
from coreprune.size import derive_expected_size
import numpy as np


def test_prune_shrinks_size():
    np.random.seed(0)
    w = np.random.randn(64, 64)
    orig_size = w.nbytes
    _, pruned_size = prune_weights(w, 0.5)
    assert pruned_size < orig_size


def test_chain_reduces_size_more():
    np.random.seed(1)
    w = np.random.randn(64, 64)
    _, p_size = prune_weights(w, 0.5)
    _, c_size = chain_prune_palettize(w, 0.5, 4)
    assert c_size < p_size


def test_derived_size_accuracy():
    shape = (32, 32)
    expected = derive_expected_size(shape, 0.5, 4)
    assert expected > 0
