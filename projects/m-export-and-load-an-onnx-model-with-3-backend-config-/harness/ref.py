import json

MODELS = [
    {"name": "encoder", "max_batch": 64, "input_shape": [1, 128], "output_shape": [1, 128]},
    {"name": "decoder", "max_batch": 32, "input_shape": [1, 32, 768], "output_shape": [1, 32, 768]}
]

def get_base_config(model_spec):
    return {
        "name": model_spec["name"],
        "platform": "onnxruntime_onnx",
        "max_batch_size": model_spec["max_batch"],
        "input": [{"name": "input_ids", "data_type": "TYPE_FP32", "dims": model_spec["input_shape"]}],
        "output": [{"name": "output", "data_type": "TYPE_FP32", "dims": model_spec["output_shape"]}]
    }

def generate_variants(model_spec):
    base = get_base_config(model_spec)
    v1 = dict(base)
    v1["instance_group"] = [{"count": 1, "kind": "KIND_GPU"}]

    v2 = dict(base)
    v2["dynamic_batching"] = {"max_queue_delay_microseconds": 5000}

    v3 = dict(base)
    v3["instance_group"] = [{"count": 2, "kind": "KIND_GPU"}]
    v3["dynamic_batching"] = {"max_queue_delay_microseconds": 2000}
    return [v1, v2, v3]

def validate_config(config):
    if not isinstance(config, dict):
        return False
    if "name" not in config or "platform" not in config:
        return False
    if config.get("max_batch_size", 0) <= 0:
        return False
    return True
