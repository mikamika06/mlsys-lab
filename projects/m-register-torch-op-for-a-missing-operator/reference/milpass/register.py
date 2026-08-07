def register_torch_op():
    return {"name": "custom::scaled_silu", "schema": "(Tensor self, float scale) -> Tensor"}
