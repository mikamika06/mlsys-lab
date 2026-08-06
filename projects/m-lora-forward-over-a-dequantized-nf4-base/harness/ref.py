import numpy as np
import re

NF4_CODEBOOK = np.array([
    -1.0, -0.6961928, -0.52507305, -0.39491749,
    -0.28444138, -0.18477343, -0.09105004, 0.0,
    0.07958029, 0.1609302, 0.2461123, 0.33791524,
    0.44070982, 0.562617, 0.72295684, 1.0
], dtype=np.float32)

SYNTHETIC_LAYOUT = [
    {"name": "model.layers.0.self_attn.q_proj", "in_features": 64, "out_features": 64},
    {"name": "model.layers.0.self_attn.k_proj", "in_features": 64, "out_features": 64},
    {"name": "model.layers.0.self_attn.v_proj", "in_features": 64, "out_features": 64},
    {"name": "model.layers.0.self_attn.o_proj", "in_features": 64, "out_features": 64},
    {"name": "model.layers.0.mlp.gate_proj", "in_features": 64, "out_features": 128},
    {"name": "model.layers.0.mlp.up_proj", "in_features": 64, "out_features": 128},
    {"name": "model.layers.0.mlp.down_proj", "in_features": 128, "out_features": 64},
]

CONFIG_TEST_CASES = [
    {
        "quant_type": "invalid_quant",
        "bnb_4bit_compute_dtype": "float16",
        "torch_dtype": "float32",
        "r": 0,
        "lora_alpha": 16,
    },
    {
        "quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
        "torch_dtype": "bfloat16",
        "r": 8,
        "lora_alpha": 16,
    },
    {
        "quant_type": "fp4",
        "bnb_4bit_compute_dtype": "float32",
        "torch_dtype": "float16",
        "r": 16,
        "lora_alpha": 64,
    },
]


def dequantize_nf4(qweight, absmax, codebook, block_size=64):
    qweight = np.asarray(qweight, dtype=np.int32)
    absmax = np.asarray(absmax, dtype=np.float32)
    codebook = np.asarray(codebook, dtype=np.float32)
    num_blocks = len(absmax)
    flattened_indices = qweight.reshape(num_blocks, block_size)
    dequantized_blocks = codebook[flattened_indices] * absmax[:, None]
    return dequantized_blocks.reshape(-1)


def lora_nf4_forward(x, qweight, absmax, codebook, lora_a, lora_b, scaling, compute_dtype="float32", block_size=64):
    target_dt = np.dtype(compute_dtype)
    x = np.asarray(x, dtype=target_dt)

    w_flat = dequantize_nf4(qweight, absmax, codebook, block_size=block_size)
    out_features, in_features = lora_b.shape[0], lora_a.shape[1]
    w_dequant = w_flat.reshape(out_features, in_features).astype(target_dt)

    base_out = x @ w_dequant.T

    lora_a = np.asarray(lora_a, dtype=target_dt)
    lora_b = np.asarray(lora_b, dtype=target_dt)

    adapter_out = (x @ lora_a.T) @ lora_b.T
    return base_out + target_dt.type(scaling) * adapter_out


def count_trainable_parameters(model_layout, target_modules, lora_r):
    total_params = 0
    trainable_params = 0

    for module in model_layout:
        name = module["name"]
        in_dim = module["in_features"]
        out_dim = module["out_features"]

        base_count = in_dim * out_dim
        total_params += base_count

        matched = False
        for target in target_modules:
            if target in name or re.search(r"\b" + re.escape(target) + r"\b", name):
                matched = True
                break

        if matched:
            adapter_count = lora_r * in_dim + lora_r * out_dim
            trainable_params += adapter_count
            total_params += adapter_count

    return {
        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
        "trainable_ratio": float(trainable_params / total_params) if total_params > 0 else 0.0,
    }


def fix_qlora_config(config):
    fixed = dict(config)

    quant_type = fixed.get("quant_type", "nf4")
    if quant_type not in ("nf4", "fp4"):
        fixed["quant_type"] = "nf4"

    bnb_4bit_compute_dtype = fixed.get("bnb_4bit_compute_dtype", "float16")
    torch_dtype = fixed.get("torch_dtype", "float16")

    if bnb_4bit_compute_dtype != torch_dtype:
        fixed["bnb_4bit_compute_dtype"] = torch_dtype
        fixed["has_dtype_mismatch_risk"] = True
    else:
        fixed["has_dtype_mismatch_risk"] = False

    r = fixed.get("r", 8)
    if r <= 0:
        fixed["r"] = 8
        r = 8

    alpha = fixed.get("lora_alpha", 16)
    fixed["scaling"] = float(alpha / r)

    return fixed
