import numpy as np
import ref


def check(workdir):
    from ortinfer.session import get_optimized_node_count, measure_latency_scaling
    model_bytes = ref.get_model_bytes()
    basic_nodes = get_optimized_node_count(model_bytes, "BASIC")
    all_nodes = get_optimized_node_count(model_bytes, "ALL")
    node_valid = 1.0 if (basic_nodes >= 1 and all_nodes >= 1 and all_nodes <= basic_nodes) else 0.0
    inputs = np.array([[1.0, -2.0, 3.0, -4.0]], dtype=np.float32)
    scaling = measure_latency_scaling(model_bytes, inputs, [1, 2])
    scaling_valid = 1.0 if (isinstance(scaling, dict) and 1 in scaling and 2 in scaling) else 0.0
    out = {
        "node_reduction_valid": node_valid,
        "scaling_valid": scaling_valid
    }
    if node_valid == 0.0:
        out["_note"] = f"node counts invalid: BASIC={basic_nodes}, ALL={all_nodes}"
    elif scaling_valid == 0.0:
        out["_note"] = "latency scaling did not return expected thread dictionary"
    return out
