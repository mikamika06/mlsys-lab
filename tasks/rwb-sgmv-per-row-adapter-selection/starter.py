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
    raise NotImplementedError('your code here')
