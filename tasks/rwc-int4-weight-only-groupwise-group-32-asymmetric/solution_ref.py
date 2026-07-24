import numpy as np


def int4_groupwise_asymmetric(W: np.ndarray, X: np.ndarray, group_size: int):
    """
    INT4 weight-only, groupwise, ASYMMETRIC quantization (the scheme behind
    torchao's Int4WeightOnly / GPTQ-style groupwise quantizers): weights are
    quantized to unsigned 4-bit codes with a separate (scale, zero) pair
    per contiguous group of `group_size` input-dim elements; activations
    stay in full precision and the matmul uses the DEQUANTIZED weight.

    W : (out_features, in_features), in_features divisible by group_size.
    X : (in_features, batch) activations.
    group_size : number of consecutive input-dim weights sharing one
        (scale, zero) pair.

    For every row of W and every group of group_size consecutive columns:
        w_min = min(group), w_max = max(group)
        scale = (w_max - w_min) / 15        # 15 == 2**4 - 1
        code  = clip(round((group - w_min) / scale), 0, 15)   # uint8
        dequant = code * scale + w_min
    (If w_max == w_min the group is constant: scale is set to 1.0 and every
    code is 0, which reconstructs the constant exactly.)

    Returns (codes, scales, zeros, output):
      codes  : (out_features, in_features) uint8, values in [0, 15].
      scales : (out_features, in_features // group_size) float64.
      zeros  : (out_features, in_features // group_size) float64 (== w_min per group).
      output : (out_features, batch) = dequant(W) @ X.
    """
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    out_f, in_f = W.shape
    groups = in_f // group_size

    codes = np.zeros((out_f, in_f), dtype=np.uint8)
    scales = np.zeros((out_f, groups), dtype=np.float64)
    zeros = np.zeros((out_f, groups), dtype=np.float64)

    for i in range(out_f):
        for g in range(groups):
            grp = W[i, g * group_size:(g + 1) * group_size]
            wmin = float(np.min(grp))
            wmax = float(np.max(grp))
            scale = (wmax - wmin) / 15.0
            if scale == 0.0:
                scale = 1.0
            code = np.clip(np.round((grp - wmin) / scale), 0, 15).astype(np.uint8)
            codes[i, g * group_size:(g + 1) * group_size] = code
            scales[i, g] = scale
            zeros[i, g] = wmin

    W_hat = np.zeros((out_f, in_f), dtype=np.float64)
    for g in range(groups):
        sl = slice(g * group_size, (g + 1) * group_size)
        W_hat[:, sl] = codes[:, sl].astype(np.float64) * scales[:, g:g + 1] + zeros[:, g:g + 1]

    output = W_hat @ X
    return codes, scales, zeros, output
