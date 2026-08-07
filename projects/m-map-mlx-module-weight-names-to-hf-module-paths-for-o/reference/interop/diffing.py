import numpy as np
import re


def normalize_gguf_key(key: str) -> str:
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


def diff_model_weights(gguf_weights: dict, convert_weights: dict, atol: float = 1e-5) -> dict:
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
