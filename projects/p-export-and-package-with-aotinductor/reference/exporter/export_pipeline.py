from typing import Any, Dict


def export_model_with_dynamic_shapes(model: Any, sample_input: Any, dynamic_shapes: Dict[str, Any]) -> Any:
    x_shape = sample_input["x"].shape
    b_min, b_max = dynamic_shapes.get("batch", (1, 128))
    s_min, s_max = dynamic_shapes.get("seq_len", (1, 512))

    batch_size = x_shape[0]
    seq_len = x_shape[1]

    if not (b_min <= batch_size <= b_max):
        raise ValueError("Batch size out of dynamic constraints")
    if not (s_min <= seq_len <= s_max):
        raise ValueError("Sequence length out of dynamic constraints")

    return {
        "model": model,
        "dynamic_constraints": {
            "batch": (b_min, b_max),
            "seq_len": (s_min, s_max)
        },
        "status": "exported"
    }
