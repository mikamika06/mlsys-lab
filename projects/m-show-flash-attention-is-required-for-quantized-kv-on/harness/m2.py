import sys
import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    out = {
        "optimal_config_matched": 0.0,
        "flash_required_proven": 0.0,
    }

    try:
        from kvquant.quant import quantize_q8_0
        from kvquant.attention import flash_attn_q8_0, unfused_attn_q8_0, optimize_context
    except Exception as e:
        out["_note"] = f"Failed to import attention module: {e}"
        return out

    try:
        candidates = ref.sample_candidates()
        best = optimize_context(candidates, recall_floor=0.90, total_budget_bytes=150 * 1024 * 1024)
        if best["num_ctx"] == 16384 and best["kv_type"] == "q8_0":
            out["optimal_config_matched"] = 1.0
        else:
            out["_note"] = f"optimize_context returned unexpected candidate: {best}"
    except Exception as e:
        out["_note"] = f"optimize_context failed: {e}"
        return out

    try:
        q, k, v = ref.generate_attn_test_data()
        k_qdict = quantize_q8_0(k, block_size=32)
        v_qdict = quantize_q8_0(v, block_size=32)

        flash_res = flash_attn_q8_0(q, k_qdict, v_qdict, sm_scale=0.125)
        unfused_res, mat_bytes = unfused_attn_q8_0(q, k_qdict, v_qdict, sm_scale=0.125)

        q_len, d_k = q.shape
        kv_len = k.shape[0]
        quantized_bytes = k_qdict["qdata"].nbytes + v_qdict["qdata"].nbytes

        if np.allclose(flash_res, unfused_res, atol=1e-4) and mat_bytes > quantized_bytes:
            out["flash_required_proven"] = 1.0
        else:
            out["_note"] = f"Flash output mismatch or memory proof failed (mat_bytes={mat_bytes}, quant_bytes={quantized_bytes})"
    except Exception as e:
        out["_note"] = f"Attention execution test failed: {e}"

    return out
