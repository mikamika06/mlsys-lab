import numpy as np
from attnsink.sink_softmax import attention_sink_softmax
from attnsink.drift import compute_drift


def make_inputs(seq_len: int = 256, d_k: int = 32, d_v: int = 32, seed: int = 42):
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((seq_len, d_k))
    K = rng.standard_normal((seq_len, d_k))
    V = rng.standard_normal((seq_len, d_v))
    K[0:4] *= 5.0
    V[0:4] *= 3.0
    return Q, K, V
