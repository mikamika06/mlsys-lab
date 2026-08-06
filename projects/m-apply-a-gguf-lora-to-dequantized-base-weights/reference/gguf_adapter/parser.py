import numpy as np


def parse_lora_gguf_and_build_delta(gguf_adapter_dict: dict, target_layer_name: str) -> dict:
    tensors = gguf_adapter_dict.get("tensors", {})
    metadata = gguf_adapter_dict.get("metadata", {})
    alpha = float(metadata.get("adapter.lora.alpha", 1.0))

    key_a = f"{target_layer_name}.lora_a"
    key_b = f"{target_layer_name}.lora_b"

    if key_a not in tensors or key_b not in tensors:
        raise KeyError(f"Target layer keys not found: {target_layer_name}")

    mat_a = np.asarray(tensors[key_a], dtype=np.float32)
    mat_b = np.asarray(tensors[key_b], dtype=np.float32)

    r = mat_a.shape[0] if mat_a.ndim == 2 else mat_a.shape[-2]
    scaling = alpha / float(r)

    delta = scaling * (mat_b @ mat_a)

    return {
        "layer_name": target_layer_name,
        "delta": delta,
        "rank": r,
        "alpha": alpha,
        "scaling": scaling
    }
