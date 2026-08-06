import numpy as np


def compute_attention(q, k, v, mask=None, backend="eager", is_causal=False):
    """Computes attention outputs for given inputs and backend."""
    head_dim = q.shape[-1]
    scale = 1.0 / np.sqrt(head_dim)
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale

    seq_len_q = q.shape[-2]
    seq_len_k = k.shape[-2]

    if is_causal:
        causal_mask = np.tril(np.ones((seq_len_q, seq_len_k), dtype=bool))
        scores = np.where(causal_mask, scores, -1e9)

    if mask is not None:
        if mask.ndim == 2:
            pad_mask = mask[:, np.newaxis, :]
        else:
            pad_mask = mask

        if backend == "buggy_right_pad":
            scores = scores + (1.0 - pad_mask) * 2.0
        else:
            scores = np.where(pad_mask > 0, scores, -1e9)

    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    return np.matmul(attn_weights, v)


def reproduce_right_padding_drift(samples, backend="eager", is_causal=True):
    """Reproduces attention numerical drift under right-padding."""
    unpadded_outputs = []
    for s in samples:
        q = s["q"][np.newaxis, :, :]
        k = s["k"][np.newaxis, :, :]
        v = s["v"][np.newaxis, :, :]
        out = compute_attention(q, k, v, mask=None, backend="eager", is_causal=is_causal)
        unpadded_outputs.append(out[0])

    batch_size = len(samples)
    max_len = max(s["q"].shape[0] for s in samples)
    head_dim = samples[0]["q"].shape[1]

    q_batch = np.zeros((batch_size, max_len, head_dim), dtype=np.float32)
    k_batch = np.zeros((batch_size, max_len, head_dim), dtype=np.float32)
    v_batch = np.zeros((batch_size, max_len, head_dim), dtype=np.float32)
    pad_mask = np.zeros((batch_size, max_len), dtype=np.float32)

    for i, s in enumerate(samples):
        l_i = s["q"].shape[0]
        q_batch[i, :l_i, :] = s["q"]
        k_batch[i, :l_i, :] = s["k"]
        v_batch[i, :l_i, :] = s["v"]
        pad_mask[i, :l_i] = 1.0

    padded_out = compute_attention(
        q_batch, k_batch, v_batch, mask=pad_mask, backend=backend, is_causal=is_causal
    )

    sample_diffs = []
    for i, s in enumerate(samples):
        l_i = s["q"].shape[0]
        diff = float(np.max(np.abs(unpadded_outputs[i] - padded_out[i, :l_i, :])))
        sample_diffs.append(diff)

    max_diff = float(max(sample_diffs))
    has_drift = max_diff > 1e-4

    return {
        "has_drift": has_drift,
        "max_diff": max_diff,
        "sample_diffs": sample_diffs,
    }
