from roofline.model import compute_prefill_flops
from roofline.predict import compute_decode_bytes


def roofline_tokens_per_sec(config: dict, batch_size: int, context_len: int, hbm_bandwidth_gbps: float, tflops: float) -> float:
    bytes_per_step = compute_decode_bytes(config, batch_size, context_len)
    hbm_bytes_per_s = hbm_bandwidth_gbps * 1e9
    max_tokens_bw = hbm_bytes_per_s / bytes_per_step if bytes_per_step > 0 else float('inf')

    layer_flops = compute_prefill_flops(config) / config["num_hidden_layers"]
    decode_flops_per_token = 2 * layer_flops
    max_tokens_compute = (tflops * 1e12) / (decode_flops_per_token * batch_size) if batch_size > 0 else float('inf')

    return float(min(max_tokens_bw * batch_size, max_tokens_compute * batch_size))
