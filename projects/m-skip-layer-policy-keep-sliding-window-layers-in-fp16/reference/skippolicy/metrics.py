import numpy as np


def quantize_dequantize_fp8(x):
    amax = np.max(np.abs(x), axis=-1, keepdims=True)
    amax = np.where(amax == 0, 1.0, amax)
    scale = 127.0 / amax
    q = np.clip(np.round(x * scale), -128, 127)
    return q / scale


def _softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def compute_ppl_delta(model_data, dtypes):
    total_loss_ref = 0.0
    total_loss_pol = 0.0
    total_tokens = 0
    for sample in model_data:
        seq_len = sample["seq_len"]
        targets = sample["targets"]
        q_list = sample["q"]
        k_list = sample["k"]
        v_list = sample["v"]
        proj = sample["proj"]
        accum_ref = np.zeros((seq_len, q_list[0].shape[-1]), dtype=np.float32)
        accum_pol = np.zeros((seq_len, q_list[0].shape[-1]), dtype=np.float32)
        for l, layer_cfg in enumerate(sample["config"]["layers"]):
            dt = dtypes.get(l, "fp16")
            k_orig = k_list[l]
            v_orig = v_list[l]
            if dt == "fp8":
                k_pol = quantize_dequantize_fp8(k_orig)
                v_pol = quantize_dequantize_fp8(v_orig)
            else:
                k_pol = k_orig
                v_pol = v_orig
            hd = k_orig.shape[-1]
            scores_ref = np.einsum("thd,shd->ths", q_list[l], k_orig) / np.sqrt(hd)
            attn_ref = _softmax(scores_ref)
            out_ref = np.einsum("ths,shd->thd", attn_ref, v_orig).mean(axis=1)
            accum_ref += out_ref
            scores_pol = np.einsum("thd,shd->ths", q_list[l], k_pol) / np.sqrt(hd)
            attn_pol = _softmax(scores_pol)
            out_pol = np.einsum("ths,shd->thd", attn_pol, v_pol).mean(axis=1)
            accum_pol += out_pol
        logits_ref = _softmax(np.matmul(accum_ref, proj))
        logits_pol = _softmax(np.matmul(accum_pol, proj))
        loss_ref = -np.log(np.maximum(1e-12, logits_ref[np.arange(seq_len), targets]))
        loss_pol = -np.log(np.maximum(1e-12, logits_pol[np.arange(seq_len), targets]))
        total_loss_ref += np.sum(loss_ref)
        total_loss_pol += np.sum(loss_pol)
        total_tokens += seq_len
    ppl_ref = np.exp(total_loss_ref / total_tokens)
    ppl_pol = np.exp(total_loss_pol / total_tokens)
    return float((ppl_pol - ppl_ref) / ppl_ref)


def passes_accuracy_gate(ppl_delta, max_allowed_delta=0.01):
    return bool(ppl_delta <= max_allowed_delta)
