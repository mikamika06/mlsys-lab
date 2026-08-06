import numpy as np


def get_valid_backends(config, env):
    res = []
    if env.get("has_flash", False) and config.get("is_decoder", True) and env.get("dtype") in ("float16", "bfloat16"):
        res.append("flash_attention_2")
    if env.get("torch_version", 0.0) >= 2.1:
        res.append("sdpa")
    res.append("eager")
    return res


def build_repro_case(batch_size, seq_len, valid_lengths):
    np.random.seed(42)
    q = np.random.randn(batch_size, seq_len, 64).astype(np.float32)
    k = np.random.randn(batch_size, seq_len, 64).astype(np.float32)
    v = np.random.randn(batch_size, seq_len, 64).astype(np.float32)
    mask = np.zeros((batch_size, seq_len), dtype=np.float32)
    for i, length in enumerate(valid_lengths):
        mask[i, :length] = 1.0
    return q, k, v, mask


def assert_parity_on_valid(q, k, v, mask, ref_fn, test_fn):
    out_ref = ref_fn(q, k, v, mask)
    out_test = test_fn(q, k, v, mask)
    diff = np.abs(out_ref - out_test)
    max_diff = 0.0
    for i in range(mask.shape[0]):
        valid_len = int(np.sum(mask[i]))
        if valid_len > 0:
            seq_diff = float(np.max(diff[i, :valid_len]))
            if seq_diff > max_diff:
                max_diff = seq_diff
    return max_diff
