import sys
import numpy as np

sys.path.insert(0, ".")
from mlxlora.eval_pitfall import LazyNode, build_lazy_lora_graph, measure_execution
from mlxlora.sizes import calculate_model_sizes
from mlxlora.verify import verify_base_weights_untouched

def test_lazy_evaluation_pitfall_and_force_eval():
    x = LazyNode("leaf", [], np.array([[1.0, 2.0]], dtype=np.float32))
    w = LazyNode("leaf", [], np.array([[0.5, -0.5], [1.0, 0.5]], dtype=np.float32))
    a = LazyNode("leaf", [], np.array([[0.1], [0.2]], dtype=np.float32))
    b = LazyNode("leaf", [], np.array([[0.3, -0.1]], dtype=np.float32))
    root = build_lazy_lora_graph(x, w, a, b, 1.0)

    unforced = measure_execution(root, force_eval=False)
    assert unforced["computed"] is False
    assert unforced["result"] is None
    assert unforced["evaluated_nodes"] < unforced["total_nodes"]

    forced = measure_execution(root, force_eval=True)
    assert forced["computed"] is True
    assert forced["result"] is not None
    assert forced["evaluated_nodes"] == forced["total_nodes"]

def test_size_calculation_and_ratios():
    model_cfg = {
        "num_layers": 2,
        "bits": 4,
        "group_size": 32,
        "scale_bytes_per_param": 2,
        "non_quantized_bytes_per_param": 2,
        "modules_per_layer": {"q_proj": [128, 128], "v_proj": [128, 128]},
        "other_modules": {"embed": [1000, 128]}
    }
    lora_cfg = {
        "r": 8,
        "target_modules": ["q_proj", "v_proj"],
        "adapter_bytes_per_param": 2,
        "safetensors_header_bytes": 512
    }
    res = calculate_model_sizes(model_cfg, lora_cfg)
    assert res["base_bytes"] > 0
    assert res["adapter_bytes"] > 0
    assert res["ratio"] < 1.0
    assert abs(res["adapter_percentage"] - (res["ratio"] * 100.0)) < 1e-5

def test_verify_detects_mutated_base_weights():
    initial = {"layer0.weight": np.array([1, 2, 3], dtype=np.int32)}
    mutated = {"layer0.weight": np.array([1, 999, 3], dtype=np.int32)}
    res = verify_base_weights_untouched(initial, mutated)
    assert res["all_frozen_matched"] is False
    assert "layer0.weight" in res["modified_keys"]
