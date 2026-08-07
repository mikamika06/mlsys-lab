import numpy as np
from attn.cache import WindowSinkKVCache
from attn.window_sink import StreamingAttentionRunner, compute_window_sink_attention


def test_sinks_are_never_evicted():
    cache = WindowSinkKVCache(num_sinks=2, window_size=3, head_dim=4)
    data_k = np.arange(20, dtype=np.float64).reshape(5, 4)
    data_v = np.arange(20, dtype=np.float64).reshape(5, 4) + 100.0

    cache.append(data_k, data_v)

    keys = cache.get_keys()
    assert len(keys) == 5
    np.testing.assert_allclose(keys[:2], data_k[:2])

    more_k = np.ones((5, 4), dtype=np.float64) * 9.0
    more_v = np.ones((5, 4), dtype=np.float64) * 99.0
    cache.append(more_k, more_v)

    assert cache.current_seq_len == 5
    keys_after = cache.get_keys()
    np.testing.assert_allclose(keys_after[:2], data_k[:2])


def test_window_boundary_transition():
    num_sinks = 2
    window_size = 4
    head_dim = 8
    np.random.seed(42)

    runner = StreamingAttentionRunner(num_sinks, window_size, head_dim)
    q_seq = np.random.randn(10, head_dim)
    k_seq = np.random.randn(10, head_dim)
    v_seq = np.random.randn(10, head_dim)

    outputs = []
    for i in range(10):
        out_i = runner.step(q_seq[i], k_seq[i], v_seq[i])
        outputs.append(out_i[0])
    outputs = np.array(outputs)

    offline_out = compute_window_sink_attention(
        q_seq, k_seq, v_seq, num_sinks=num_sinks, window_size=window_size
    )

    np.testing.assert_allclose(outputs, offline_out, atol=1e-6)
