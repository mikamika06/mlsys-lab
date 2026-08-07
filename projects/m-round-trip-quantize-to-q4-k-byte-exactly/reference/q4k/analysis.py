import numpy as np
from q4k.quant import dequantize_q4_k_superblock, quantize_q4_k_superblock


def find_worst_subblocks(weights: np.ndarray, data: bytes) -> list[int]:
    w = weights.flatten()
    dec = dequantize_q4_k_superblock(data)
    errors = (w - dec) ** 2
    sub_mse = []
    for i in range(16):
        sub_err = np.mean(errors[i * 16:(i + 1) * 16])
        sub_mse.append((sub_err, i))
    sub_mse.sort(key=lambda x: x[0], reverse=True)
    return [idx for _, idx in sub_mse]


def compare_q4k_q40_mse(weights: np.ndarray) -> dict[str, float]:
    w = weights.flatten()
    q_k_bytes = quantize_q4_k_superblock(w)
    dec_k = dequantize_q4_k_superblock(q_k_bytes)
    mse_k = float(np.mean((w - dec_k) ** 2))
    mse_0 = 0.0
    for i in range(8):
        sub = w[i * 32:(i + 1) * 32]
        mx = np.max(np.abs(sub))
        d = mx / 7.0 if mx > 0 else 1.0
        q = np.clip(np.round(sub / d) + 8, 0, 15)
        dec_sub = (q - 8) * d
        mse_0 += float(np.mean((sub - dec_sub) ** 2))
    mse_0 /= 8.0
    return {"q4_k_mse": mse_k, "q4_0_mse": mse_0, "ratio": mse_k / (mse_0 if mse_0 > 0 else 1e-6)}
