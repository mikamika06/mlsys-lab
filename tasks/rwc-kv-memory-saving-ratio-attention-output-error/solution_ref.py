import numpy as np

def kv_memory_saving_ratio_and_attention_error(
    kv_fp16: np.ndarray,
    kv_fp8:  np.ndarray,
    query:   np.ndarray
) -> tuple[float, float]:
    # ratio of bytes
    ratio = kv_fp16.nbytes / kv_fp8.nbytes

    # dequantise fp8 to float32 in [-1, 1]
    kv_fp8_flt = kv_fp8.astype(np.float32) / 127.0

    K_fp16, V_fp16 = kv_fp16[0], kv_fp16[1]
    K_fp8 , V_fp8  = kv_fp8_flt[0], kv_fp8_flt[1]

    d = K_fp16.shape[-1]
    scale = np.sqrt(d)

    # FP16 attention
    scores_fp16 = query @ K_fp16.T / scale
    exp_fp16 = np.exp(scores_fp16 - np.max(scores_fp16, axis=-1, keepdims=True))
    softmax_fp16 = exp_fp16 / np.sum(exp_fp16, axis=-1, keepdims=True)
    out_fp16 = softmax_fp16 @ V_fp16

    # FP8 attention
    scores_fp8 = query @ K_fp8.T / scale
    exp_fp8 = np.exp(scores_fp8 - np.max(scores_fp8, axis=-1, keepdims=True))
    softmax_fp8 = exp_fp8 / np.sum(exp_fp8, axis=-1, keepdims=True)
    out_fp8 = softmax_fp8 @ V_fp8

    error = np.max(np.abs(out_fp16 - out_fp8))

    return ratio, error
