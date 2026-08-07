import numpy as np
from kvquant.quant import dequantize_q8_0


def flash_attn_q8_0(
    q: np.ndarray,
    k_qdict: dict,
    v_qdict: dict,
    sm_scale: float = 1.0,
    block_size: int = 32,
) -> np.ndarray:
    """Fused flash attention over quantized KV without full dequant materialization."""
    q_len, d_k = q.shape
    k_qdata = k_qdict["qdata"]
    k_scales = k_qdict["scales"]
    v_qdata = v_qdict["qdata"]
    v_scales = v_qdict["scales"]
    kv_len = k_qdict["orig_shape"][0]

    out = np.zeros((q_len, d_k), dtype=np.float32)

    for i in range(q_len):
        q_i = q[i]
        max_score = -np.inf
        sum_exp = 0.0
        acc = np.zeros(d_k, dtype=np.float32)

        blocks_per_row = d_k // block_size
        for j in range(kv_len):
            k_row = np.zeros(d_k, dtype=np.float32)
            for b in range(blocks_per_row):
                idx = j * blocks_per_row + b
                k_row[b * block_size : (b + 1) * block_size] = (
                    k_qdata[idx].astype(np.float32) * k_scales[idx]
                )

            score = np.dot(q_i, k_row) * sm_scale

            v_row = np.zeros(d_k, dtype=np.float32)
            for b in range(blocks_per_row):
                idx = j * blocks_per_row + b
                v_row[b * block_size : (b + 1) * block_size] = (
                    v_qdata[idx].astype(np.float32) * v_scales[idx]
                )

            if score > max_score:
                alpha = np.exp(max_score - score) if max_score != -np.inf else 0.0
                max_score = score
                sum_exp = sum_exp * alpha + 1.0
                acc = acc * alpha + v_row
            else:
                alpha = np.exp(score - max_score)
                sum_exp += alpha
                acc += alpha * v_row

        out[i] = acc / sum_exp

    return out


def unfused_attn_q8_0(
    q: np.ndarray,
    k_qdict: dict,
    v_qdict: dict,
    sm_scale: float = 1.0,
) -> tuple[np.ndarray, int]:
    """Unfused attention requiring full dequantization materialization in global memory."""
    k = dequantize_q8_0(k_qdict)
    v = dequantize_q8_0(v_qdict)

    materialized_bytes = k.nbytes + v.nbytes + (q.shape[0] * k.shape[0] * 4)

    scores = np.matmul(q, k.T) * sm_scale
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    out = np.matmul(attn_weights, v)

    return out, materialized_bytes


def optimize_context(
    candidates: list[dict], recall_floor: float, total_budget_bytes: int
) -> dict:
    """Select candidate with max num_ctx satisfying recall floor and memory budget."""
    valid = []
    for c in candidates:
        if c["recall"] >= recall_floor and c["memory_bytes"] <= total_budget_bytes:
            valid.append(c)

    if not valid:
        raise ValueError("No configuration meets both recall floor and memory budget.")

    valid.sort(key=lambda x: (x["num_ctx"], -x["memory_bytes"]), reverse=True)
    return valid[0]
