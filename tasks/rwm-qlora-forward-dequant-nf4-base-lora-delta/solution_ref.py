NF4_LEVELS = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
]


def qlora_forward(
    x: list[list[float]],
    nf4_codes: list[list[int]],
    absmax: list[list[float]],
    blocksize: int,
    A: list[list[float]],
    B: list[list[float]],
    alpha: float,
) -> list[list[float]]:
    """
    QLoRA forward pass: dequantize the NF4 base weight (blockwise absmax),
    add the scaled LoRA delta B@A, then apply the resulting linear layer.
    """
    d_out = len(nf4_codes)
    d_in = len(nf4_codes[0])
    n_blocks = d_in // blocksize

    w_dq = []
    for o in range(d_out):
        row_dq = []
        for j in range(d_in):
            b_idx = j // blocksize
            val = NF4_LEVELS[nf4_codes[o][j]] * absmax[o][b_idx]
            row_dq.append(val)
        w_dq.append(row_dq)

    r = len(A)
    scaling = float(alpha) / r

    delta = []
    for o in range(d_out):
        row_delta = []
        for j in range(d_in):
            s = 0.0
            for k in range(r):
                s += B[o][k] * A[k][j]
            row_delta.append(scaling * s)
        delta.append(row_delta)

    w_eff = []
    for o in range(d_out):
        eff_row = []
        for j in range(d_in):
            eff_row.append(w_dq[o][j] + delta[o][j])
        w_eff.append(eff_row)

    n = len(x)
    y = []
    for i in range(n):
        y_row = []
        for o in range(d_out):
            s = 0.0
            for j in range(d_in):
                s += x[i][j] * w_eff[o][j]
            y_row.append(s)
        y.append(y_row)

    return y
