def sgmv_apply(x: list[list[float]], adapter_id: list[int],
                A_bank: list[list[list[float]]], B_bank: list[list[list[float]]], scale: list[float]) -> list[list[float]]:
    """SGMV (Segmented Gather Matrix-Vector multiply): apply a PER-ROW
    LoRA adapter, selected from a shared bank, to a mixed batch (the
    core primitive behind multi-LoRA serving systems like S-LoRA /
    Punica -- many concurrent requests, each using a different adapter,
    batched into one GEMM-friendly call).

    x          : (N, d_in) input rows.
    adapter_id : (N,) int, row i uses adapter `adapter_id[i]`.
    A_bank     : (num_adapters, d_in, r) -- all adapters share rank r.
    B_bank     : (num_adapters, r, d_out).
    scale      : (num_adapters,) float, per-adapter LoRA scale.

    Row i's output is
        scale[adapter_id[i]] * (x[i] @ A_bank[adapter_id[i]]) @ B_bank[adapter_id[i]]

    Returns (N, d_out).
    """
    N = len(x)
    d_in = len(x[0]) if N > 0 else 0
    d_out = len(B_bank[0][0]) if len(B_bank) > 0 and len(B_bank[0]) > 0 else 0
    r = len(A_bank[0][0]) if len(A_bank) > 0 and len(A_bank[0]) > 0 else 0

    out = [[0.0 for _ in range(d_out)] for _ in range(N)]

    for i in range(N):
        aid = int(adapter_id[i])

        inter_vec = [0.0] * r
        for k in range(r):
            acc = 0.0
            for j in range(d_in):
                acc += x[i][j] * A_bank[aid][j][k]
            inter_vec[k] = acc

        delta = [0.0] * d_out
        for k in range(d_out):
            acc = 0.0
            for j in range(r):
                acc += inter_vec[j] * B_bank[aid][j][k]
            delta[k] = acc

        s = scale[aid]
        for k in range(d_out):
            out[i][k] = s * delta[k]

    return out
