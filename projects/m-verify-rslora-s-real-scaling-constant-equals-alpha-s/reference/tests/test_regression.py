import sys
sys.path.insert(0, ".")
import numpy as np
from adapters.verify import verify_rslora_scaling
from adapters.compare import compare_magnitudes, compare_parameters

def test_rslora_scaling_is_not_plain_lora():
    r = 64
    alpha = 32.0
    rslora_val = verify_rslora_scaling(r, alpha, "rslora")
    plain_val = alpha / r
    assert not np.isclose(rslora_val, plain_val), "rsLoRA scaling incorrectly equals plain LoRA scaling"
    assert np.isclose(rslora_val, alpha / np.sqrt(r))

def test_parameter_counts_valid():
    p_lora, p_ia3 = compare_parameters(64, 64, 8)
    assert p_lora > p_ia3, "LoRA parameters should exceed IA3 parameters for rank 8"
    assert p_ia3 == 64
