import sys
sys.path.insert(0, ".")
from extractor.parser import extract_stop_sequences
from extractor.adapter import apply_adapter_and_forward
import numpy as np


def test_stop_sequences():
    cfg = {"eos_token_id": 2, "stop_strings": ["<|end|>"]}
    res = extract_stop_sequences(cfg)
    assert 2 in res
    assert "<|end|>" in res


def test_adapter_perturbation():
    w = [[1.0, 0.0], [0.0, 1.0]]
    adapter = ([[0.5, 0.0], [0.0, 0.5]], [[1.0, 1.0], [1.0, 1.0]])
    x = [1.0, 1.0]
    out_base = x @ np.array(w)
    out_adapted = apply_adapter_and_forward(w, adapter, x)
    assert not np.allclose(out_base, out_adapted)
