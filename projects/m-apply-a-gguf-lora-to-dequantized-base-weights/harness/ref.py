import numpy as np


def generate_peft_data(seed=42):
    rng = np.random.default_rng(seed)
    layers = ["layers.0.self_attn.q_proj", "layers.0.self_attn.v_proj"]
    peft_dict = {}
    shapes = {}
    r = 8
    alpha = 32.0
    for l_name in layers:
        in_dim, out_dim = 16, 32
        shapes[l_name] = (out_dim, in_dim)
        peft_dict[f"base_model.model.{l_name}.lora_A.weight.default"] = rng.standard_normal((r, in_dim)).astype(np.float32)
        peft_dict[f"base_model.model.{l_name}.lora_B.weight.default"] = rng.standard_normal((out_dim, r)).astype(np.float32)
    return peft_dict, alpha, shapes


def ref_convert_peft_to_gguf(peft_dict, alpha):
    gguf_dict = {
        "metadata": {"adapter.lora.alpha": float(alpha)},
        "tensors": {}
    }
    for k, v in peft_dict.items():
        clean = k.replace("base_model.model.", "").replace(".default", "")
        if ".lora_A.weight" in clean:
            gk = clean.replace(".lora_A.weight", ".lora_a")
        elif ".lora_B.weight" in clean:
            gk = clean.replace(".lora_B.weight", ".lora_b")
        else:
            gk = clean
        gguf_dict["tensors"][gk] = np.asarray(v, dtype=np.float32)
    return gguf_dict


def ref_parse_and_build_delta(gguf_dict, target_layer):
    tensors = gguf_dict["tensors"]
    alpha = float(gguf_dict["metadata"]["adapter.lora.alpha"])
    mat_a = tensors[f"{target_layer}.lora_a"]
    mat_b = tensors[f"{target_layer}.lora_b"]
    r = mat_a.shape[0]
    scaling = alpha / float(r)
    delta = scaling * (mat_b @ mat_a)
    return {
        "layer_name": target_layer,
        "delta": delta,
        "rank": r,
        "alpha": alpha,
        "scaling": scaling
    }


def ref_apply_lora(dequantized_base, delta_dict):
    out = {}
    for k, v in dequantized_base.items():
        if k in delta_dict:
            out[k] = np.asarray(v, dtype=np.float32) + np.asarray(delta_dict[k]["delta"], dtype=np.float32)
        else:
            out[k] = np.asarray(v, dtype=np.float32).copy()
    return out
