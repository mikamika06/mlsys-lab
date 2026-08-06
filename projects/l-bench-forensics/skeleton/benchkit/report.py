def prefill_decode_split(rows):
    """prefill_rows, decode_rows, best_prefill_ts, best_decode_ts,
    prefill_decode_ratio."""
    raise NotImplementedError


def noisiest(rows, limit=3):
    raise NotImplementedError


def pick_ubatch(rows, min_decode_ts=0.0):
    """{"options": [...], "chosen": ubatch}.

    Best prefill throughput among the micro-batches whose decode still clears
    the floor. Every option carries its own numbers and a meets_floor flag, so
    the choice can be argued with rather than trusted.
    """
    raise NotImplementedError


def model_summary(rows):
    """Per model: size_bytes, params, best throughputs, and the weight bytes
    and parameters a second of decoding implies."""
    raise NotImplementedError
