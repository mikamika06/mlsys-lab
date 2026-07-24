import numpy as np


def kv_scale_granularity_delta(Q, K, V):
    """
    Q: (H, M, D), K, V: (H, N, D) float arrays.

    Run attention on the original Q/K/V in float64 as the reference
    O_ref. Quantize-dequantize K and V to simulated e4m3 (scale =
    amax/448) two ways -- once with a single per-tensor scale, once with
    an independent per-head scale -- run attention with each
    reconstruction, and compute the per-head-averaged relative error
    against O_ref for each:

      delta = mean_h( ||O_quant[h] - O_ref[h]|| / ||O_ref[h]|| )

    Returns (per_tensor_delta, per_head_delta) as Python floats.
    """
    raise NotImplementedError('your code here')
