import re

TREE_A = {
    "model.layers.0.self_attn.q_proj": (4096, 4096),
    "model.layers.0.self_attn.k_proj": (4096, 4096),
    "model.layers.0.self_attn.v_proj": (4096, 4096),
    "model.layers.0.self_attn.o_proj": (4096, 4096),
    "model.layers.0.mlp.gate_proj": (11008, 4096),
    "model.layers.0.mlp.up_proj": (11008, 4096),
    "model.layers.0.mlp.down_proj": (4096, 11008),
    "model.layers.1.self_attn.q_proj": (4096, 4096),
    "model.layers.1.self_attn.k_proj": (4096, 4096),
    "model.layers.1.self_attn.v_proj": (4096, 4096),
    "model.layers.1.self_attn.o_proj": (4096, 4096),
    "model.layers.1.mlp.gate_proj": (11008, 4096),
    "model.layers.1.mlp.up_proj": (11008, 4096),
    "model.layers.1.mlp.down_proj": (4096, 11008),
    "lm_head": (32000, 4096),
}

TREE_B = {
    "encoder.block.0.layer.0.SelfAttention.q": (768, 768),
    "encoder.block.0.layer.0.SelfAttention.k": (768, 768),
    "encoder.block.0.layer.0.SelfAttention.v": (768, 768),
    "encoder.block.0.layer.0.SelfAttention.o": (768, 768),
    "encoder.block.0.layer.1.DenseReluDense.wi": (3072, 768),
    "encoder.block.0.layer.1.DenseReluDense.wo": (768, 3072),
    "encoder.block.1.layer.0.SelfAttention.q": (768, 768),
    "encoder.block.1.layer.0.SelfAttention.k": (768, 768),
    "encoder.block.1.layer.0.SelfAttention.v": (768, 768),
    "encoder.block.1.layer.0.SelfAttention.o": (768, 768),
    "encoder.block.1.layer.1.DenseReluDense.wi": (3072, 768),
    "encoder.block.1.layer.1.DenseReluDense.wo": (768, 3072),
}

CONFIGS = [
    (TREE_A, ".*(q_proj|v_proj)$", ["q_proj", "v_proj"]),
    (TREE_A, ".*self_attn\\.(q|k|v|o)_proj$", ["q_proj", "k_proj", "v_proj", "o_proj"]),
    (TREE_B, ".*SelfAttention\\.(q|v)$", ["q", "v"]),
    (TREE_B, ".*DenseReluDense\\.wi$", ["wi"]),
]


def resolve_by_suffix(named_modules, suffixes):
    matched = []
    for name in sorted(named_modules.keys()):
        for sfx in suffixes:
            if name == sfx or name.endswith("." + sfx):
                matched.append(name)
                break
    return sorted(matched)


def resolve_by_regex(named_modules, pattern):
    compiled = re.compile(pattern)
    matched = [name for name in sorted(named_modules.keys()) if compiled.search(name)]
    return sorted(matched)


def compute_param_count(named_modules, module_names):
    total = 0
    for name in module_names:
        if name in named_modules:
            shape = named_modules[name]
            total += shape[0] * shape[1]
    return total


def verify_equivalence(named_modules, pattern, suffixes):
    reg_set = set(resolve_by_regex(named_modules, pattern))
    sfx_set = set(resolve_by_suffix(named_modules, suffixes))
    return reg_set == sfx_set
