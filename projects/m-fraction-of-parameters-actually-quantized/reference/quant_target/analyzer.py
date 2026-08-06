from quant_target.targeting import filter_target_modules
from quant_target.metrics import compute_quantized_fraction


def analyze_model_quantization(config, ignore_list=None):
    targets = filter_target_modules(config, ignore_list=ignore_list)
    fraction, head_cost = compute_quantized_fraction(config, targets)
    return {
        "targets": targets,
        "fraction": fraction,
        "head_cost": head_cost,
        "total_params": sum(l.get("params", 0) for l in config.get("layers", []))
    }
