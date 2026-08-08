BITS_PER_FTYPE = {
    "F32": 32.0,
    "F16": 16.0,
    "Q8_0": 8.5,
    "Q4_K": 4.5,
}

CONFIGS = [
    {
        "tensors": [
            {"name": "blk.0.attn_q.weight", "shape": [4096, 4096], "importance": 0.9},
            {"name": "blk.0.attn_k.weight", "shape": [4096, 1024], "importance": 0.8},
            {"name": "blk.0.attn_v.weight", "shape": [4096, 1024], "importance": 0.8},
            {"name": "blk.0.attn_output.weight", "shape": [4096, 4096], "importance": 0.85},
            {"name": "blk.0.attn_norm.weight", "shape": [4096], "importance": 1.0},
            {"name": "blk.0.ffn_gate.weight", "shape": [11008, 4096], "importance": 0.7},
            {"name": "blk.0.ffn_up.weight", "shape": [11008, 4096], "importance": 0.7},
            {"name": "blk.0.ffn_down.weight", "shape": [4096, 11008], "importance": 0.75},
            {"name": "blk.0.ffn_norm.weight", "shape": [4096], "importance": 1.0},
            {"name": "output_norm.weight", "shape": [4096], "importance": 1.0},
        ],
        "budget_bytes": 120000000,
    },
    {
        "tensors": [
            {"name": "token_embd.weight", "shape": [32000, 4096], "importance": 0.95},
            {"name": "blk.0.attn_qkv.weight", "shape": [6144, 4096], "importance": 0.85},
            {"name": "blk.0.attn_qkv.bias", "shape": [6144], "importance": 1.0},
            {"name": "blk.0.attn_norm.scale", "shape": [4096], "importance": 1.0},
            {"name": "output.weight", "shape": [32000, 4096], "importance": 0.9},
        ],
        "budget_bytes": 220000000,
    },
]


def tensor_elements(shape):
    p = 1
    for s in shape:
        p *= s
    return p


def tensor_bytes(shape, ftype):
    el = tensor_elements(shape)
    if len(shape) == 1:
        return el * 4
    bits = BITS_PER_FTYPE.get(ftype, 16.0)
    return int((el * bits + 7) // 8)


def recipe_bytes(config, recipe):
    tot = 0
    for t in config["tensors"]:
        ftype = recipe.get(t["name"], "F16")
        tot += tensor_bytes(t["shape"], ftype)
    return tot


def solve_recipe(config, budget_bytes):
    recipe = {}
    for t in config["tensors"]:
        if len(t["shape"]) == 1:
            recipe[t["name"]] = "F32"

    matrix_tensors = [t for t in config["tensors"] if len(t["shape"]) > 1]
    sorted_matrices = sorted(matrix_tensors, key=lambda x: x.get("importance", 0.5), reverse=True)

    for t in sorted_matrices:
        recipe[t["name"]] = "Q4_K"

    candidates = ["F32", "F16", "Q8_0", "Q4_K"]

    for t in sorted_matrices:
        for ftype in candidates:
            recipe[t["name"]] = ftype
            if recipe_bytes(config, recipe) <= budget_bytes:
                break
        else:
            recipe[t["name"]] = "Q4_K"

    return recipe


def verify_f32_1d(config, recipe):
    for t in config["tensors"]:
        if len(t["shape"]) == 1:
            if recipe.get(t["name"]) != "F32":
                return False
    return True
