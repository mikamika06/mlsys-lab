import json
import re
import struct
import numpy as np

np.random.seed(42)

MLX_BLOCK_0 = {
    "attention.wq.weight": np.random.randn(64, 64).astype(np.float32),
    "attention.wk.weight": np.random.randn(64, 64).astype(np.float32),
    "attention.wv.weight": np.random.randn(64, 64).astype(np.float32),
    "attention.wo.weight": np.random.randn(64, 64).astype(np.float32),
    "feed_forward.w1.weight": np.random.randn(256, 64).astype(np.float32),
    "feed_forward.w2.weight": np.random.randn(64, 256).astype(np.float32),
    "feed_forward.w3.weight": np.random.randn(256, 64).astype(np.float32),
    "attention_norm.weight": np.ones((64,), dtype=np.float32),
    "ffn_norm.weight": np.ones((64,), dtype=np.float32),
}

GGUF_WEIGHTS = {
    "blk.0.attn_q.weight": np.random.randn(32, 32).astype(np.float32),
    "blk.0.attn_k.weight": np.random.randn(32, 32).astype(np.float32),
    "blk.0.attn_v.weight": np.random.randn(32, 32).astype(np.float32),
    "blk.0.attn_output.weight": np.random.randn(32, 32).astype(np.float32),
    "blk.0.ffn_gate.weight": np.random.randn(128, 32).astype(np.float32),
    "blk.0.ffn_down.weight": np.random.randn(32, 128).astype(np.float32),
    "blk.0.ffn_up.weight": np.random.randn(128, 32).astype(np.float32),
    "blk.0.attn_norm.weight": np.ones((32,), dtype=np.float32),
    "blk.0.ffn_norm.weight": np.ones((32,), dtype=np.float32),
    "token_embd.weight": np.random.randn(100, 32).astype(np.float32),
    "output_norm.weight": np.ones((32,), dtype=np.float32),
    "output.weight": np.random.randn(100, 32).astype(np.float32),
}

CONVERT_WEIGHTS = {
    "model.layers.0.self_attn.q_proj.weight": GGUF_WEIGHTS["blk.0.attn_q.weight"].copy(),
    "model.layers.0.self_attn.k_proj.weight": GGUF_WEIGHTS["blk.0.attn_k.weight"].copy(),
    "model.layers.0.self_attn.v_proj.weight": GGUF_WEIGHTS["blk.0.attn_v.weight"].copy(),
    "model.layers.0.self_attn.o_proj.weight": GGUF_WEIGHTS["blk.0.attn_output.weight"].copy(),
    "model.layers.0.mlp.gate_proj.weight": GGUF_WEIGHTS["blk.0.ffn_gate.weight"].copy(),
    "model.layers.0.mlp.down_proj.weight": GGUF_WEIGHTS["blk.0.ffn_down.weight"].copy(),
    "model.layers.0.mlp.up_proj.weight": GGUF_WEIGHTS["blk.0.ffn_up.weight"].copy(),
    "model.layers.0.input_layernorm.weight": GGUF_WEIGHTS["blk.0.attn_norm.weight"].copy(),
    "model.layers.0.post_attention_layernorm.weight": GGUF_WEIGHTS["blk.0.ffn_norm.weight"].copy(),
    "model.embed_tokens.weight": GGUF_WEIGHTS["token_embd.weight"].copy(),
    "model.norm.weight": GGUF_WEIGHTS["output_norm.weight"].copy(),
    "lm_head.weight": GGUF_WEIGHTS["output.weight"].copy(),
}
CONVERT_WEIGHTS["model.layers.0.self_attn.q_proj.weight"] += 1e-6

TINY_TENSOR_SET = {
    "embedding": np.random.randn(16, 32).astype(np.float32),
    "proj": np.random.randn(32, 16).astype(np.float16),
    "bias": np.zeros((16,), dtype=np.int32),
}


def map_mlx_block_weights(mlx_weights, block_idx):
    mapped = {}
    mapping_rules = {
        "attention.wq": f"model.layers.{block_idx}.self_attn.q_proj",
        "attention.wk": f"model.layers.{block_idx}.self_attn.k_proj",
        "attention.wv": f"model.layers.{block_idx}.self_attn.v_proj",
        "attention.wo": f"model.layers.{block_idx}.self_attn.o_proj",
        "feed_forward.w1": f"model.layers.{block_idx}.mlp.gate_proj",
        "feed_forward.w2": f"model.layers.{block_idx}.mlp.down_proj",
        "feed_forward.w3": f"model.layers.{block_idx}.mlp.up_proj",
        "attention_norm": f"model.layers.{block_idx}.input_layernorm",
        "ffn_norm": f"model.layers.{block_idx}.post_attention_layernorm",
    }
    for key, val in mlx_weights.items():
        clean_key = key
        for prefix in (f"model.layers.{block_idx}.", f"layers.{block_idx}.", f"{block_idx}."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
                break
        matched = False
        for prefix, hf_prefix in mapping_rules.items():
            if clean_key == prefix:
                mapped[hf_prefix] = val
                matched = True
                break
            elif clean_key.startswith(prefix + "."):
                suffix = clean_key[len(prefix):]
                mapped[hf_prefix + suffix] = val
                matched = True
                break
        if not matched:
            mapped[key] = val
    return mapped


def normalize_gguf_key(key):
    m = re.match(r"^blk\.(\d+)\.(.+)$", key)
    if m:
        idx = m.group(1)
        sub = m.group(2)
        gguf_rules = {
            "attn_q": "self_attn.q_proj",
            "attn_k": "self_attn.k_proj",
            "attn_v": "self_attn.v_proj",
            "attn_output": "self_attn.o_proj",
            "ffn_gate": "mlp.gate_proj",
            "ffn_down": "mlp.down_proj",
            "ffn_up": "mlp.up_proj",
            "attn_norm": "input_layernorm",
            "ffn_norm": "post_attention_layernorm",
        }
        for g_prefix, hf_prefix in gguf_rules.items():
            if sub == g_prefix or sub.startswith(g_prefix + "."):
                suffix = sub[len(g_prefix):]
                return f"model.layers.{idx}.{hf_prefix}{suffix}"
    global_rules = {
        "token_embd.weight": "model.embed_tokens.weight",
        "output_norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
    }
    return global_rules.get(key, key)


def diff_model_weights(gguf_weights, convert_weights, atol=1e-5):
    norm_gguf = {normalize_gguf_key(k): v for k, v in gguf_weights.items()}
    common_keys = sorted(set(norm_gguf.keys()) & set(convert_weights.keys()))
    mismatches = []
    matched_keys = 0
    max_abs_diff = 0.0

    for k in common_keys:
        a = norm_gguf[k]
        b = convert_weights[k]
        if a.shape != b.shape:
            mismatches.append(k)
            continue
        diff = float(np.max(np.abs(a - b)))
        if diff > max_abs_diff:
            max_abs_diff = diff
        if diff <= atol:
            matched_keys += 1
        else:
            mismatches.append(k)

    return {
        "common_keys": len(common_keys),
        "matched_keys": matched_keys,
        "max_abs_diff": max_abs_diff,
        "mismatches": mismatches,
    }


def compute_safetensors_header(tensors):
    dtype_map = {
        np.dtype("float32"): "F32",
        np.dtype("float16"): "F16",
        np.dtype("float64"): "F64",
        np.dtype("int32"): "I32",
        np.dtype("int64"): "I64",
        np.dtype("int16"): "I16",
        np.dtype("int8"): "I8",
        np.dtype("uint8"): "U8",
        np.dtype("bool"): "BOOL",
    }
    header = {}
    offset = 0
    for name in sorted(tensors.keys()):
        arr = tensors[name]
        size = arr.nbytes
        dt = dtype_map.get(arr.dtype, "F32")
        header[name] = {
            "data_offsets": [offset, offset + size],
            "dtype": dt,
            "shape": list(arr.shape),
        }
        offset += size

    header_json = json.dumps(header, separators=(",", ":"), sort_keys=True)
    header_bytes = header_json.encode("utf-8")
    header_len = len(header_bytes)
    prefix_bytes = struct.pack("<Q", header_len)
    return header_len, prefix_bytes
