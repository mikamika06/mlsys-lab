import numpy as np
from attnsink.sink_softmax import attention_sink_softmax


def test_sink_overlap_no_double_count():
    rng = np.random.default_rng(42)
    Q = rng.standard_normal((10, 16))
    K = rng.standard_normal((10, 16))
    V = rng.standard_normal((10, 16))

    sink_size = 4
    window_size = 10

    out_sink, lse_sink = attention_sink_softmax(Q, K, V, sink_size=sink_size, window_size=window_size)
    out_nosink, lse_nosink = attention_sink_softmax(Q, K, V, sink_size=0, window_size=window_size)

    assert np.allclose(out_sink, out_nosink, atol=1e-7)
    assert np.allclose(lse_sink, lse_nosink, atol=1e-7)
