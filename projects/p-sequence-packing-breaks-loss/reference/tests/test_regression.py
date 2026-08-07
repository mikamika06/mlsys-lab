import numpy as np
from seqpack.attention import create_block_diagonal_mask, compute_packed_attention
from seqpack.loss import compute_packed_loss
from seqpack.pack import measure_attention_leakage


def test_block_diagonal_mask_prevents_leakage():
    seq_ids = np.array([0, 0, 1, 1], dtype=np.int64)
    mask = create_block_diagonal_mask(seq_ids)
    assert not mask[2, 0]
    assert not mask[3, 1]


def test_packed_attention_zero_leakage():
    seq_ids = np.array([0, 0, 1, 1], dtype=np.int64)
    Q = np.ones((4, 4))
    K = np.ones((4, 4))
    V = np.ones((4, 4))
    _, weights = compute_packed_attention(Q, K, V, seq_ids)
    leakage = measure_attention_leakage(weights, seq_ids)
    assert leakage == 0.0


def test_loss_normalization_ignores_pad():
    logits = np.ones((10, 5))
    labels = np.array([0, 1, 2, 3, 4, 0, 0, 0, 0, 0], dtype=np.int64)
    label_mask = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    seq_ids = np.array([0, 0, 0, 1, 1, -1, -1, -1, -1, -1], dtype=np.int64)

    loss = compute_packed_loss(logits, labels, label_mask, seq_ids)
    expected = -np.log(0.2)
    assert abs(loss - expected) < 1e-4
