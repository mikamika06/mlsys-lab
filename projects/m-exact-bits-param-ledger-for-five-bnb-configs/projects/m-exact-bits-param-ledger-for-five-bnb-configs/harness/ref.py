import numpy as np

CONFIGS = [
    {"name": "fp4_standard", "bits_per_weight": 4, "double_quant": False, "quant_type": "fp4"},
    {"name": "nf4_standard", "bits_per_weight": 4, "double_quant": False, "quant_type": "nf4"},
    {"name": "nf4_double", "bits_per_weight": 4, "double_quant": True, "quant_type": "nf4", "nested_bits": 8, "block_size": 256},
    {"name": "int8_standard", "bits_per_weight": 8, "double_quant": False, "quant_type": "int8"},
    {"name": "int4_double", "bits_per_weight": 4, "double_quant": True, "quant_type": "int4", "nested_bits": 8, "block_size": 64}
]

def get_ledger(config):
    bpw = config["bits_per_weight"]
    if config["double_quant"]:
        nb = config.get("nested_bits", 8)
        bs = config.get("block_size", 256)
        overhead = nb / bs
        total_bits = bpw + overhead
    else:
        total_bits = float(bpw)
    return {"bits_per_param": total_bits, "config_name": config["name"]}

def predict_memory_footprint(config, num_params):
    ledger = get_ledger(config)
    bits = ledger["bits_per_param"]
    return int(np.ceil(num_params * bits / 8.0))

def nested_absmax_quantize(tensor, block_size=256):
    flat = tensor.astype(np.float64)
    n = len(flat)
    padded_len = ((n + block_size - 1) // block_size) * block_size
    padded = np.pad(flat, (0, padded_len - n), mode='constant')
    blocks = padded.reshape(-1, block_size)
    absmaxs = np.max(np.abs(blocks), axis=1)
    absmaxs_scale = np.max(np.abs(absmaxs))
    if absmaxs_scale == 0:
        absmaxs_quantized = np.zeros_like(absmaxs)
        scale2 = 1.0
    else:
        scale2 = absmaxs_scale / 127.0
        absmaxs_quantized = np.round(absmaxs / scale2).clip(-127, 127)
    absmaxs_dequant = absmaxs_quantized * scale2
    quantized_blocks = np.zeros_like(blocks)
    for i in range(len(blocks)):
        scale1 = absmaxs_dequant[i]
        if scale1 == 0:
            quantized_blocks[i] = 0
        else:
            q = np.round(blocks[i] / (scale1 / 7.0)).clip(-7, 7)
            quantized_blocks[i] = q * (scale1 / 7.0)
    dequant = quantized_blocks.flatten()[:n]
    return dequant
