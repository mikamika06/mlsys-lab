def compute_hbm_utilization(batch_size,
                            seq_len,
                            hidden_dim,
                            ff_hidden_mult=4):
    """
    Compute the arithmetic intensity (FLOPs / bytes) of a single transformer decoder step.
    Parameters are all integers; the result is returned as a float.
    """
    b, s, h, f = batch_size, seq_len, hidden_dim, ff_hidden_mult
    flops = 6 * b * s * h ** 2 + 2 * b * s * f * h ** 2 + 2 * b * s ** 2 * h
    bytes_ = 8 * b * s * h * (1 + 3 + 2 * f)
    return float(flops / bytes_)
