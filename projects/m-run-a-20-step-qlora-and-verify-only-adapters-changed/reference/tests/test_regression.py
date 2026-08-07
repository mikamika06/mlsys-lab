import sys
sys.path.insert(0, ".")
import numpy as np
from qlora.trainer import run_qlora_steps
from qlora.verify import verify_adapters_changed

def test_base_weights_frozen():
    np.random.seed(42)
    init_model = {
        "base_weight": np.random.randn(16, 16).astype(np.float32),
        "lora_a": np.zeros((4, 16), dtype=np.float32),
        "lora_b": np.zeros((16, 4), dtype=np.float32)
    }
    data = [np.random.randn(16).astype(np.float32) for _ in range(25)]
    final_model = run_qlora_steps(init_model, data, steps=20)
    assert verify_adapters_changed(init_model, final_model) is True
    assert np.allclose(init_model["base_weight"], final_model["base_weight"], atol=1e-7)
