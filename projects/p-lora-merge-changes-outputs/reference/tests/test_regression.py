import sys
sys.path.insert(0, ".")
from lora_merge.merger import LoRAMerger
import numpy as np


def test_merge_preserves_outputs():
    np.random.seed(42)
    w = [np.random.randn(16, 16)]
    a = [np.random.randn(4, 16)]
    b = [np.random.randn(16, 4)]
    merger = LoRAMerger(w, a, b, alpha=16.0, rank=4)
    merger.safe_merge()
    x = np.random.randn(16, 1)
    err = merger.evaluate_prompts([x])
    assert err < 1e-5, f"Error too high: {err}"


def test_scaling_factor():
    merger = LoRAMerger([np.zeros((4, 4))], [np.zeros((2, 4))], [np.zeros((4, 2))], alpha=8.0, rank=4)
    assert merger.verify_scaling() is True
