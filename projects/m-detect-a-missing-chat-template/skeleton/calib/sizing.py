def select_calibration_shape(target_tokens, max_seqlen, min_seqlen, mem_limit_mb, bytes_per_token):
    """
    Selects (N, seqlen) minimizing calibration compute while covering target_tokens.
    """
    raise NotImplementedError
