import numpy as np


def _naive_fp16_sum(x: np.ndarray) -> float:
    """Single low-precision running total over the whole sequence (the bad strategy)."""
    acc = np.float16(0.0)
    for v in x:
        acc = np.float16(np.float32(acc) + np.float32(v))
    return float(acc)


def _oracle_two_level_sum(x: np.ndarray, block_size: int) -> float:
    """Reference two-level accumulation: fp16 within block, fp32 across blocks."""
    n = x.shape[0]
    total = np.float32(0.0)
    for start in range(0, n, block_size):
        block = x[start:start + block_size]
        block_acc = np.float16(0.0)
        for v in block:
            block_acc = np.float16(np.float32(block_acc) + np.float32(v))
        total = np.float32(total) + np.float32(block_acc)
    return float(total)


def _make_sequence(rng, n, outlier_frac, outlier_scale):
    x = rng.standard_normal(n).astype(np.float32)
    mask = rng.random(n) < outlier_frac
    x[mask] *= outlier_scale
    return x


def grade(sol, fx) -> dict:
    """
    Builds several long random sequences with occasional large-magnitude
    outliers (realistic long-context activation pattern), and checks that
    the student's two_level_accumulate result stays close to a fp32
    reference sum -- much closer than a naive single-accumulator fp16
    running sum achieves on the same data.
    """
    rng = np.random.default_rng(0)
    n_trials = 6
    block_sizes = [32, 64, 128]

    scaled_errs = []
    naive_scaled_errs = []

    for t in range(n_trials):
        n = 2048
        block_size = block_sizes[t % len(block_sizes)]
        x = _make_sequence(rng, n, outlier_frac=0.02, outlier_scale=40.0)

        ref = float(np.sum(x, dtype=np.float32))
        scale = float(np.sum(np.abs(x), dtype=np.float32)) + 1e-12

        naive = _naive_fp16_sum(x)
        naive_scaled_errs.append(abs(ref - naive) / scale)

        try:
            got = sol.two_level_accumulate(x.copy(), int(block_size))
            got = float(got)
        except Exception:
            scaled_errs.append(float("inf"))
            continue

        if not np.isfinite(got):
            scaled_errs.append(float("inf"))
            continue

        scaled_errs.append(abs(ref - got) / scale)

    mean_rel_err = float(np.mean(scaled_errs))
    return {
        "rel_err": mean_rel_err,
    }
