import numpy as np

FP4_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)


def ref_compute_max_scale(tensor: np.ndarray, max_bound: float = 127.0) -> float:
    if tensor.size == 0:
        return 1.0
    amax = float(np.max(np.abs(tensor)))
    if amax == 0.0:
        return 1.0
    return amax / float(max_bound)


def ref_compute_entropy_scale(
    tensor: np.ndarray,
    num_bins: int = 2048,
    num_quant_steps: int = 128,
    max_bound: float = 127.0,
) -> float:
    abs_tensor = np.abs(tensor.astype(np.float64)).flatten()
    max_val = float(np.max(abs_tensor)) if abs_tensor.size > 0 else 0.0
    if max_val == 0.0:
        return 1.0

    hist, bin_edges = np.histogram(abs_tensor, bins=num_bins, range=(0.0, max_val))
    hist = hist.astype(np.float64)

    best_kl = float("inf")
    best_threshold = max_val

    start_bin = max(1, num_quant_steps)

    for threshold_bin in range(start_bin, num_bins + 1):
        p_hist = hist[:threshold_bin].copy()
        outliers = np.sum(hist[threshold_bin:])
        p_hist[-1] += outliers

        if np.sum(p_hist) == 0:
            continue

        p = p_hist / np.sum(p_hist)

        quant_bin_size = threshold_bin / float(num_quant_steps)
        q = np.zeros(threshold_bin, dtype=np.float64)

        for i in range(num_quant_steps):
            start = int(np.floor(i * quant_bin_size))
            end = int(np.ceil((i + 1) * quant_bin_size))
            end = min(end, threshold_bin)
            if start < end:
                q[start:end] += np.sum(p[start:end])

        for i in range(num_quant_steps):
            start = int(np.floor(i * quant_bin_size))
            end = int(np.ceil((i + 1) * quant_bin_size))
            end = min(end, threshold_bin)
            count = end - start
            if count > 0 and q[start] > 0:
                q[start:end] = q[start] / count

        eps = 1e-12
        p = np.where(p == 0, eps, p)
        q = np.where(q == 0, eps, q)

        p = p / np.sum(p)
        q = q / np.sum(q)

        kl_div = np.sum(p * np.log(p / q))

        if kl_div < best_kl:
            best_kl = kl_div
            best_threshold = bin_edges[threshold_bin]

    return float(best_threshold / max_bound)


def ref_nvfp4_round_trip(tensor: np.ndarray, block_size: int = 16) -> np.ndarray:
    shape = tensor.shape
    flat = tensor.astype(np.float64).flatten()
    n = flat.size
    pad_len = (block_size - (n % block_size)) % block_size
    if pad_len > 0:
        flat = np.pad(flat, (0, pad_len), mode="constant", constant_values=0.0)

    num_blocks = flat.size // block_size
    reshaped = flat.reshape(num_blocks, block_size)

    block_max = np.max(np.abs(reshaped), axis=1)
    scales = block_max / 6.0
    scales = np.where(scales == 0.0, 1.0, scales)

    scaled_reshaped = reshaped / scales[:, None]
    signs = np.sign(scaled_reshaped)
    signs = np.where(signs == 0, 1.0, signs)
    abs_scaled = np.abs(scaled_reshaped)

    diffs = np.abs(abs_scaled[:, :, None] - FP4_VALUES[None, None, :])
    mag_indices = np.argmin(diffs, axis=-1)

    codes = (np.where(signs < 0, 8, 0) | mag_indices).astype(np.uint8)

    reshaped_codes = codes.reshape(num_blocks, block_size)
    sign_bits = (reshaped_codes >> 3) & 1
    mag_indices_out = reshaped_codes & 7

    mags = FP4_VALUES[mag_indices_out]
    signs_out = np.where(sign_bits == 1, -1.0, 1.0)
    dequant = signs_out * mags * scales[:, None]

    return dequant.reshape(-1)[:n].reshape(shape)


def generate_test_tensors():
    np.random.seed(1337)
    tensors = [
        np.random.normal(loc=0.0, scale=1.0, size=(128, 256)),
        np.random.exponential(scale=2.0, size=(512,)),
        np.random.uniform(-100.0, 100.0, size=(64, 64)),
    ]
    return tensors
