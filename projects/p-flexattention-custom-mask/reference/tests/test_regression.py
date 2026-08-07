import numpy as np
from mask_engine.predicate import make_mask_predicate
from mask_engine.block import build_block_mask
from mask_engine.compile import compile_attention_mask

def test_predicate_basic():
    doc_ids = [0, 0, 0, 1, 1]
    pred = make_mask_predicate(doc_ids, window_size=2)
    assert pred(2, 0) == True
    assert pred(4, 3) == True
    assert pred(3, 0) == False

def test_block_mask_shape():
    doc_ids = [0] * 16
    pred = make_mask_predicate(doc_ids, window_size=4)
    bm = build_block_mask(16, 4, pred)
    assert bm.shape == (4, 4)

def test_compiled_mask_identity():
    doc_ids = [0, 0, 1, 1]
    pred = make_mask_predicate(doc_ids, window_size=2)
    m = compile_attention_mask(pred, 4)
    assert m.shape == (4, 4)
