import fnmatch

CONFIGS = [
    {
        "modules": [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.mlp.gate_proj",
        ],
        "targets": ["*self_attn*"],
        "ignores": ["*q_proj"],
    },
    {
        "modules": [
            "model.layers.1.block_sparse_moe.gate",
            "model.layers.1.block_sparse_moe.experts.0.w1",
        ],
        "targets": ["*block_sparse_moe*"],
        "ignores": ["*.gate"],
    },
    {
        "modules": [
            "model.layers.2.mlp.gate_proj",
            "model.layers.2.mlp.up_proj",
        ],
        "targets": ["*mlp*"],
        "ignores": ["*up_proj"],
    },
]

MOE_STRUCTURES = [
    {
        "modules": [
            "model.layers.0.block_sparse_moe.gate.weight",
            "model.layers.0.block_sparse_moe.experts.0.w1.weight",
            "model.layers.0.input_layernorm.weight",
        ]
    },
    {
        "modules": [
            "model.layers.1.moe.router.weight",
            "model.layers.1.moe.mlp.down_proj.weight",
            "model.layers.1.post_attention_layernorm.weight",
        ]
    },
]

ROUTER_STATES = [
    {
        "model.layers.0.block_sparse_moe.gate": "int4",
        "model.layers.0.mlp.gate_proj": "fp16",
    },
    {
        "model.layers.1.moe.router": "fp32",
        "model.layers.1.mlp.down_proj": "int8",
    },
]


def resolve_targets(modules, targets, ignores):
    matched = []
    for m in modules:
        is_target = any(fnmatch.fnmatch(m, t) for t in targets)
        is_ignore = any(fnmatch.fnmatch(m, i) for i in ignores)
        if is_target and not is_ignore:
            matched.append(m)
    return sorted(matched)


def build_moe_ignore_list(model_structure):
    ignores = []
    for name in model_structure.get("modules", []):
        if "gate" in name or "router" in name or "norm" in name:
            ignores.append(name)
    return sorted(list(set(ignores)))


def find_wrong_router(quantized_modules):
    wrong = []
    for name, dtype in quantized_modules.items():
        if ("gate" in name or "router" in name) and dtype != "fp32":
            wrong.append(name)
    return sorted(wrong)
