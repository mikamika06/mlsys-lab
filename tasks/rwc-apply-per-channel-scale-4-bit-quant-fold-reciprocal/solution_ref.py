import math


def _group_quant_dequant(x: list[list[float]], bits: int) -> list[list[float]]:
    """Uniform affine min-max quantizer/dequantizer along the LAST axis:
    every group of `group_size` values along that axis shares one
    scale/zero-point."""
    qmax = (1 << bits) - 1
    d0 = len(x)
    d1 = len(x[0])
    out = [[0.0] * d1 for _ in range(d0)]

    for i in range(d0):
        xmin = x[i][0]
        xmax = x[i][0]
        for k in range(1, d1):
            val = x[i][k]
            if val < xmin:
                xmin = val
            if val > xmax:
                xmax = val

        scale = (xmax - xmin) / qmax
        if scale == 0:
            scale = 1.0

        zero_point = round(-xmin / scale)

        for k in range(d1):
            q = round(x[i][k] / scale + zero_point)
            if q < 0:
                q = 0
            elif q > qmax:
                q = qmax
            out[i][k] = (q - zero_point) * scale

    return out


def awq_apply_fixed_scale(
    W: list[list[float]], s: list[float], X: list[list[float]], group_size: int, bits: int = 4
) -> list[list[float]]:
    """
    W: (out_features, in_features) weight matrix.
    s: (in_features,) positive per-input-channel AWQ smoothing scale
        (already chosen -- this function just applies it).
    X: (batch, in_features) activations.
    group_size: number of consecutive input channels sharing one
        quantizer scale/zero-point (in_features must be a multiple of
        group_size).
    bits: quantizer bit-width.

    AWQ transform for a fixed scale:
      1. Ws = W * s (scale weight COLUMNS, i.e. input channels, up by s)
      2. Wq = fake-quantize Ws to `bits` bits, GROUP-WISE along the
         input-channel axis: for each output row and each contiguous
         block of `group_size` input channels, use a uniform affine
         min-max quantizer/dequantizer over just that block.
      3. Xs = X / s (fold the reciprocal scale into activations)
      4. output = Xs @ Wq.T

    Steps 1 and 3 cancel exactly in real arithmetic; the whole point is
    that step 2's rounding error is smaller after up-scaling channels
    that would otherwise be crushed by outlier-dominated group ranges.

    Returns `output`, shape (batch, out_features).
    """
    out_features = len(W)
    in_features = len(W[0])

    Ws = [[0.0] * in_features for _ in range(out_features)]
    for i in range(out_features):
        for j in range(in_features):
            Ws[i][j] = W[i][j] * s[j]

    num_groups = in_features // group_size
    Ws_flattened = []
    for i in range(out_features):
        for g in range(num_groups):
            group = []
            for j in range(group_size):
                group.append(Ws[i][g * group_size + j])
            Ws_flattened.append(group)

    Wq_flattened = _group_quant_dequant(Ws_flattened, bits)

    Wq = [[0.0] * in_features for _ in range(out_features)]
    flat_idx = 0
    for i in range(out_features):
        for g in range(num_groups):
            for j in range(group_size):
                Wq[i][g * group_size + j] = Wq_flattened[flat_idx][j]
            flat_idx += 1

    batch_size = len(X)
    Xs = [[0.0] * in_features for _ in range(batch_size)]
    for i in range(batch_size):
        for j in range(in_features):
            Xs[i][j] = X[i][j] / s[j]

    output = [[0.0] * out_features for _ in range(batch_size)]
    for i in range(batch_size):
        for j in range(out_features):
            acc = 0.0
            for k in range(in_features):
                acc += Xs[i][k] * Wq[j][k]
            output[i][j] = acc

    return output
