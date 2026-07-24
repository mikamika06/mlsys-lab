import numpy as np

RATIOS = tuple(i / 10 for i in range(11))  # 0.0, 0.1, ..., 1.0


def awq_ratio_search(W: np.ndarray, X: np.ndarray, n_bits: int = 4):
    """
    Search the fixed AWQ ratio grid RATIOS = (0.0, 0.1, ..., 1.0) for the
    ratio that minimizes the calibration-activation-weighted output MSE
    after quantizing the (scaled) weights to `n_bits`.

    For each ratio r in RATIOS:
      s_x = mean(|X|, axis=0)              # per-input-channel activation scale
      s = s_x ** r
      s = s / sqrt(s.max() * s.min())      # keep dynamic range balanced
      W_scaled = W * s[None, :]
      W_hat = dequantize(quantize_per_row(W_scaled, n_bits)) / s[None, :]
      mse = mean((X @ W.T - X @ W_hat.T) ** 2)

    where quantize_per_row does symmetric round-to-nearest quantization to
    `n_bits` independently for each output row (using that row's own
    max-abs scale).

    Returns (best_ratio_index, best_mse): the index into RATIOS achieving
    the smallest mse, and that mse value.
    """
    raise NotImplementedError('your code here')
