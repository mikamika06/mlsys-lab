import numpy as np

E2M1_TABLE = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=np.float32)


def generate_test_blocks(seed=42):
    rng = np.random.default_rng(seed)
    blocks = []
    scales = [120, 127, 130, 135, 140]
    for s in scales:
        nibbles = rng.integers(0, 16, size=32, dtype=np.uint8)
        blocks.append((s, nibbles))
    return blocks


def generate_continuous_blocks(seed=123):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=(10, 32)).astype(np.float32)


def ref_decode_mxfp4_block(scale_e8m0: int, nibbles: np.ndarray) -> np.ndarray:
    scale_e8m0 = int(scale_e8m0) & 0xFF
    if scale_e8m0 == 0:
        scale = 0.0
    else:
        scale = 2.0 ** (scale_e8m0 - 127)
    nibbles_arr = np.asarray(nibbles, dtype=np.uint8) & 0x0F
    return (E2M1_TABLE[nibbles_arr] * scale).astype(np.float32)


def ref_enumerate_mxfp4_grid(scale_e8m0: int) -> np.ndarray:
    nibbles = np.arange(16, dtype=np.uint8)
    vals = ref_decode_mxfp4_block(scale_e8m0, nibbles)
    return np.sort(vals)


def ref_mxfp4_vs_q4_0_crossover(blocks: np.ndarray) -> dict[str, float]:
    blocks = np.asarray(blocks, dtype=np.float32)
    n_blocks = blocks.shape[0]

    mxfp4_mse_sum = 0.0
    q4_0_mse_sum = 0.0

    for i in range(n_blocks):
        orig = blocks[i]
        max_val = np.max(np.abs(orig))
        if max_val == 0:
            scale_e8m0 = 0
        else:
            unbiased_exp = np.ceil(np.log2(max_val / 6.0))
            scale_e8m0 = int(np.clip(unbiased_exp + 127, 0, 255))

        grid = ref_enumerate_mxfp4_grid(scale_e8m0)
        idx = np.argmin(np.abs(orig[:, None] - grid[None, :]), axis=1)
        mxfp4_rec = grid[idx]
        mxfp4_mse_sum += float(np.mean((orig - mxfp4_rec) ** 2))

        max_abs = np.max(np.abs(orig))
        if max_abs == 0.0:
            q4_0_rec = np.zeros_like(orig)
        else:
            scale = max_abs / 7.0
            q_signed = np.clip(np.round(orig / scale), -8, 7)
            q4_0_rec = q_signed * scale

        q4_0_mse_sum += float(np.mean((orig - q4_0_rec) ** 2))

    mxfp4_avg_mse = mxfp4_mse_sum / n_blocks
    q4_0_avg_mse = q4_0_mse_sum / n_blocks
    return {
        "mxfp4_avg_mse": float(mxfp4_avg_mse),
        "q4_0_avg_mse": float(q4_0_avg_mse),
        "mxfp4_wins": 1.0 if mxfp4_avg_mse < q4_0_avg_mse else 0.0
    }
