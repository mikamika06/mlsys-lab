def per_channel_vs_per_token_k_mse(K: list[list[float]], bits: int) -> list[float]:
    """
    K: (n_tokens, d_channels) fp64 key cache. bits: quantizer bit-width.

    Quantize K two ways with a uniform affine (asymmetric) min-max
    quantizer:
      - per-channel: one scale/zero-point per COLUMN, min/max taken
        across all tokens.
      - per-token: one scale/zero-point per ROW, min/max taken across
        all channels.

    Returns [mse_per_channel, mse_per_token], the reconstruction
    MSE (mean squared error of dequant(quant(K)) vs K) of each scheme.
    """
    raise NotImplementedError('your code here')
