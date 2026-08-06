import numpy as np
from mxfp4.decode import decode_mxfp4_block, quantize_q4_0_block, decode_q4_0_block


def enumerate_mxfp4_grid(scale_e8m0: int) -> np.ndarray:
    """Return all 16 unique FP32 values representable by MXFP4 for a given scale_e8m0."""
    nibbles = np.arange(16, dtype=np.uint8)
    vals = decode_mxfp4_block(scale_e8m0, nibbles)
    return np.sort(vals)


def mxfp4_vs_q4_0_crossover(blocks: np.ndarray) -> dict[str, float]:
    """Compare reconstruction MSE between MXFP4 and Q4_0 across input blocks."""
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

        grid = enumerate_mxfp4_grid(scale_e8m0)
        idx = np.argmin(np.abs(orig[:, None] - grid[None, :]), axis=1)

        e2m1_unscaled = np.array([
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
        ], dtype=np.float32)

        mxfp4_nibbles = np.zeros(32, dtype=np.uint8)
        for k in range(32):
            val = grid[idx[k]]
            matches = np.where(np.abs(e2m1_unscaled * (2.0 ** (scale_e8m0 - 127) if scale_e8m0 > 0 else 0.0) - val) < 1e-7)[0]
            mxfp4_nibbles[k] = matches[0] if len(matches) > 0 else 0

        mxfp4_rec = decode_mxfp4_block(scale_e8m0, mxfp4_nibbles)
        mxfp4_mse_sum += float(np.mean((orig - mxfp4_rec) ** 2))

        q_scale, q_nibbles = quantize_q4_0_block(orig)
        q4_0_rec = decode_q4_0_block(q_scale, q_nibbles)
        q4_0_mse_sum += float(np.mean((orig - q4_0_rec) ** 2))

    mxfp4_avg_mse = mxfp4_mse_sum / n_blocks
    q4_0_avg_mse = q4_0_mse_sum / n_blocks

    return {
        "mxfp4_avg_mse": float(mxfp4_avg_mse),
        "q4_0_avg_mse": float(q4_0_avg_mse),
        "mxfp4_wins": 1.0 if mxfp4_avg_mse < q4_0_avg_mse else 0.0
    }
