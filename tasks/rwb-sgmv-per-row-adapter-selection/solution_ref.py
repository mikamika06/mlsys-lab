import numpy as np


def sgmv_apply(x: np.ndarray, adapter_id: np.ndarray,
                A_bank: np.ndarray, B_bank: np.ndarray, scale: np.ndarray) -> np.ndarray:
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
    x = np.asarray(x, dtype=np.float64)
    adapter_id = np.asarray(adapter_id, dtype=np.int64)
    A_bank = np.asarray(A_bank, dtype=np.float64)
    B_bank = np.asarray(B_bank, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)

    N = x.shape[0]
    d_in = x.shape[1]
    d_out = B_bank.shape[2]
    r = A_bank.shape[2]

    out = np.empty((N, d_out), dtype=np.float64)

    for i in range(N):
        aid = int(adapter_id[i])
        
        intermediate = 0.0
        # x[i] @ A_bank[aid]: shape (r,)
        # x[i] is (d_in,), A_bank[aid] is (d_in, r)
        inter_vec = np.empty((r,), dtype=np.float64)
        for k in range(r):
            acc = 0.0
            for j in range(d_in):
                acc += x[i, j] * A_bank[aid, j, k]
            inter_vec[k] = acc

        # inter_vec @ B_bank[aid]: shape (d_out,)
        # inter_vec is (r,), B_bank[aid] is (r, d_out)
        delta = np.empty((d_out,), dtype=np.float64)
        for k in range(d_out):
            acc = 0.0
            for j in range(r):
                acc += inter_vec[j] * B_bank[aid, j, k]
            delta[k] = acc

        s = scale[aid]
        for k in range(d_out):
            out[i, k] = s * delta[k]

    return out
