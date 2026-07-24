import numpy as np


def compare_k_quant_granularity(K: np.ndarray, bits: int):
    """
    Compare per-channel vs per-token symmetric quantization of a key cache
    K (n_tokens, d_channels).

    Returns (mse_per_channel, mse_per_token):
      mse_per_channel: MSE when each COLUMN (channel) gets its own scale,
                        calibrated over all tokens.
      mse_per_token:   MSE when each ROW (token) gets its own scale,
                        calibrated over all channels.
    """
    raise NotImplementedError('your code here')
