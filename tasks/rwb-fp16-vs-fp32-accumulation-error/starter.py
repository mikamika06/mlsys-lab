import numpy as np


def fp16_vs_fp32_attention_error(Q, K, V):
    """
    Q: (n, d), K: (m, d), V: (m, d_v) float arrays.

    Compute exact float64 attention O_ref, then attention run with every
    accumulation step (score dot product, softmax normalizer sum, output
    dot product) rounded to float16 at each step (O_16), and the same
    with float32 (O_32). Do not use np.dot/@/np.sum for these three
    reductions -- they may accumulate internally in a wider dtype.

    Returns (fp16_rel_err, fp32_rel_err): relative L2 error of O_16 and
    O_32 against O_ref, as Python floats.
    """
    raise NotImplementedError('your code here')
