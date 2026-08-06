import time
import numpy as np
from bnb_quant.loader import load_model_weights, quantize_fp16_to_int8, pack_prequantized_artifact


def measure_loading_latency(raw_weights):
    """Compares loading time between inflight quantization and pre-quantized loading."""
    t0 = time.perf_counter()
    for _ in range(10):
        _ = load_model_weights(raw_weights, mode="inflight")
    t1 = time.perf_counter()
    inflight_time = (t1 - t0) / 10.0

    artifacts = {k: pack_prequantized_artifact(v) for k, v in raw_weights.items()}

    t2 = time.perf_counter()
    for _ in range(10):
        loaded = {}
        for k, art in artifacts.items():
            loaded[k] = {"qweight": art["qweight"].copy(), "scales": art["scales"].copy()}
    t3 = time.perf_counter()
    prequant_time = (t3 - t2) / 10.0

    ratio = inflight_time / max(prequant_time, 1e-9)
    return {
        "inflight_sec": inflight_time,
        "prequant_sec": prequant_time,
        "ratio": ratio
    }


def measure_memory_footprint(raw_weights):
    """Measures raw weight bytes vs quantized byte representation."""
    fp16_bytes = sum(w.nbytes for w in raw_weights.values())

    quant_bytes = 0
    for w in raw_weights.values():
        q_w, scales = quantize_fp16_to_int8(w)
        quant_bytes += q_w.nbytes + scales.nbytes

    return {
        "fp16_bytes": fp16_bytes,
        "quant_bytes": quant_bytes,
        "bytes_saved_ratio": float(fp16_bytes) / float(quant_bytes)
    }
