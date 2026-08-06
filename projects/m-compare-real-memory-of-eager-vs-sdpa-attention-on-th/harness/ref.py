CONFIGS = [
    {"hidden_size": 64, "num_heads": 4},
    {"hidden_size": 128, "num_heads": 8},
    {"hidden_size": 32, "num_heads": 2},
]


def build_model(config):
    from reference.attnmem.model import TinyAttentionModel
    return TinyAttentionModel(config)


def compute_ref_ratio(model, inputs):
    from reference.attnmem.measure import compute_size_ratio
    return compute_size_ratio(model, inputs)
