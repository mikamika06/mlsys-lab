"""Roofline predictor for decode throughput."""


def predict_decode_throughput(config: dict, batch_size: int, context_len: int, peak_tflops: float, hbm_bw_gbps: float, dtype_bytes: int = 2) -> dict:
    """Predict decode performance characteristics under hardware bounds."""
    raise NotImplementedError
