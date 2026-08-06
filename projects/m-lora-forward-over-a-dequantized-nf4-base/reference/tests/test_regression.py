import numpy as np
import sys
sys.path.insert(0, ".")

from qlora.forward import lora_nf4_forward
from qlora.config import fix_qlora_config


def test_qlora_dtype_consistency():
    qweight = np.array([0, 1, 2, 3], dtype=np.int32)
    absmax = np.array([1.0], dtype=np.float32)
    codebook = np.array([-1.0, -0.5, 0.5, 1.0], dtype=np.float32)
    lora_a = np.array([[1.0, 0.5]], dtype=np.float32)
    lora_b = np.array([[0.5], [1.0]], dtype=np.float32)
    x = np.array([[1.0, 2.0]], dtype=np.float32)

    out = lora_nf4_forward(x, qweight, absmax, codebook, lora_a, lora_b, scaling=2.0, compute_dtype="float32", block_size=4)
    assert not np.isnan(out).any(), "Forward result should not contain NaNs"
    assert out.dtype == np.float32, f"Expected float32, got {out.dtype}"


def test_config_fix_resolves_mismatch():
    bad_cfg = {
        "quant_type": "nf4",
        "bnb_4bit_compute_dtype": "float16",
        "torch_dtype": "float32",
        "r": 16,
        "lora_alpha": 32,
    }
    fixed = fix_qlora_config(bad_cfg)
    assert fixed["bnb_4bit_compute_dtype"] == fixed["torch_dtype"], "Config repair must align compute dtypes"
    assert fixed["has_dtype_mismatch_risk"] is True, "Must flag mismatch risk prior to repair"
