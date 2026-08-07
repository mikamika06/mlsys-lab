CONFIGS = [
    {"total_weight_params": 7000000000},
    {"total_weight_params": 13000000000},
    {"total_weight_params": 70000000000},
]

SCHEMES = [
    {"name": "w4a16", "bits": 4, "group_size": 128, "meta_overhead_factor": 1.02},
    {"name": "w8a16", "bits": 8, "group_size": 128, "meta_overhead_factor": 1.01},
]

ARCHITECTURES = ["ampere", "hopper", "blackwell"]


def estimate_checkpoint_size(config, scheme):
    from compress.predictor import estimate_checkpoint_size as fn
    return fn(config, scheme)


def get_supported_schemes(arch):
    from compress.chooser import get_supported_schemes as fn
    return fn(arch)
