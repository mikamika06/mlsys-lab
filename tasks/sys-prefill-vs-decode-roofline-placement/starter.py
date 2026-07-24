def prefill_decode_roofline(
    hidden_size: int,
    num_heads: int,
    prefill_len: int,
    dtype_bytes: int,
    peak_flops_per_s: float,
    peak_bytes_per_s: float,
) -> dict:
    """Modeled FLOPs, HBM bytes and arithmetic intensity for one transformer
    layer, for (a) a length-``prefill_len`` prefill pass and (b) one decode
    step immediately after that prefill (so its KV cache holds
    ``prefill_len`` prior tokens).

    Per-layer weight scalars: QKV (3*H^2) + output proj (H^2) + FFN
    (H->4H->H, 8*H^2) = 12*H^2, each ``dtype_bytes`` bytes.

    Prefill (``P`` tokens, one batched forward pass; attention cost uses
    the full (non-causal) P x P score matrix as a simplifying model):
        flops  = 12*H^2*P + 2*H*P^2
        bytes  = 12*H^2*dtype_bytes            (weights, loaded once)
               + dtype_bytes*P*H                (read the P input embeddings)
               + dtype_bytes*2*P*H              (write the P new K/V rows)

    Decode (1 new token, S = ``prefill_len`` cached tokens attended to):
        flops  = 12*H^2 + 2*H*S
        bytes  = 12*H^2*dtype_bytes            (weights, loaded again)
               + dtype_bytes*2*S*H              (read the cached K/V)
               + dtype_bytes*H                  (read the new input token)
               + dtype_bytes*2*H                (write the new token's K/V)

    A phase is classified ``"compute-bound"`` if its arithmetic intensity
    (flops/bytes) is at or above the hardware ridge point
    ``peak_flops_per_s / peak_bytes_per_s``, else ``"bandwidth-bound"``.

    Returns
    -------
    dict with keys "prefill" and "decode", each mapping to a dict with
    keys "flops", "bytes", "ai", "roofline_class".
    """
    raise NotImplementedError('your code here')
