import numpy as np
from sparse_eval.checkpoint import get_checkpoint_stats, load_sparse_checkpoint, save_sparse_checkpoint


def test_checkpoint_compression_ratio():
    rng = np.random.RandomState(42)
    mat = np.zeros((32, 64), dtype=np.float32)
    for r in range(32):
        for c in range(16):
            mat[r, c * 4] = float(rng.randn())
            mat[r, c * 4 + 1] = float(rng.randn())

    stats = get_checkpoint_stats(mat)
    assert stats["compression_ratio"] < 0.7, f"Compression ratio {stats['compression_ratio']} >= 0.7"
    assert stats["sparse_bytes"] < stats["dense_bytes"], "Sparse checkpoint is not smaller than dense"


def test_checkpoint_roundtrip_integrity():
    rng = np.random.RandomState(42)
    mat = np.zeros((32, 64), dtype=np.float32)
    for r in range(32):
        for c in range(16):
            mat[r, c * 4] = float(rng.randn())
            mat[r, c * 4 + 2] = float(rng.randn())

    ckpt = save_sparse_checkpoint(mat)
    assert ckpt["format"] == "2:4_structured", f"Invalid format {ckpt.get('format')}"
    rec = load_sparse_checkpoint(ckpt)
    assert np.max(np.abs(mat - rec)) < 1e-5, "Reconstructed matrix does not match original"
