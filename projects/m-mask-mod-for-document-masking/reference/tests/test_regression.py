import sys
sys.path.insert(0, ".")
from docmask.masks import document_mask_mod, prefix_lm_mask_mod
from docmask.sparsity import block_sparsity_fraction
import numpy as np

def test_document_mask_boundaries():
    doc_ids = np.array([[0, 0, 1, 1]])
    mask = document_mask_mod(doc_ids)
    assert mask(0, 0, 0, 1) is True or mask(0, 0, 0, 1) == True
    assert mask(0, 0, 0, 2) is False or mask(0, 0, 0, 2) == False

def test_prefix_lm_behavior():
    prefix_lengths = [2]
    mask = prefix_lm_mask_mod(prefix_lengths)
    assert mask(0, 0, 3, 0) is True or mask(0, 0, 3, 0) == True
    assert mask(0, 0, 2, 3) is False or mask(0, 0, 2, 3) == False

def test_sparsity_bounds():
    def trivial_mask(b, h, q, kv):
        return q >= kv
    frac = block_sparsity_fraction(trivial_mask, 256, 128)
    assert 0.0 <= frac <= 1.0
