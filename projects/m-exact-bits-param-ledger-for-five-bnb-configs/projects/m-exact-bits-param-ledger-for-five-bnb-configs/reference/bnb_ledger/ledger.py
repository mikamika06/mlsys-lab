import numpy as np

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
