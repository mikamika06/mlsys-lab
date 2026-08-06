import math
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
    batch = X.shape[1]

    codes = np.zeros((out_f, in_f), dtype=np.uint8)
    scales = np.zeros((out_f, groups), dtype=np.float64)
    zeros = np.zeros((out_f, groups), dtype=np.float64)

    for i in range(out_f):
        for g in range(groups):
            start = g * group_size
            wmin = W[i, start]
            wmax = W[i, start]
            for j in range(1, group_size):
                val = W[i, start + j]
                if val < wmin:
                    wmin = val
                if val > wmax:
                    wmax = val

            scale = (wmax - wmin) / 15.0
            if scale == 0.0:
                scale = 1.0

            scales[i, g] = scale
            zeros[i, g] = wmin

            for j in range(group_size):
                val = W[i, start + j]
                rounded = round((val - wmin) / scale)
                if rounded < 0:
                    c = 0
                elif rounded > 15:
                    c = 15
                else:
                    c = int(rounded)
                codes[i, start + j] = c

    W_hat = np.zeros((out_f, in_f), dtype=np.float64)
    for i in range(out_f):
        for g in range(groups):
            start = g * group_size
            scale = scales[i, g]
            wmin = zeros[i, g]
            for j in range(group_size):
                W_hat[i, start + j] = float(codes[i, start + j]) * scale + wmin

    output = np.zeros((out_f, batch), dtype=np.float64)
    for i in range(out_f):
        for b in range(batch):
            acc = 0.0
            for k in range(in_f):
                acc += W_hat[i, k] * X[k, b]
            output[i, b] = acc

    return codes, scales, zeros, output
