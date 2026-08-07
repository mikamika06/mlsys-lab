INVENTORIES = [
    {
        "tensors": [
            {"name": "blk.0.attn_q.weight", "shape": [512, 512], "qtype": "Q4_K"},
            {"name": "blk.0.attn_k.weight", "shape": [512, 512], "qtype": "Q4_K"},
            {"name": "output.weight", "shape": [1024, 512], "qtype": "F32"}
        ]
    },
    {
        "tensors": [
            {"name": "blk.0.ffn_gate.weight", "shape": [1024, 512], "qtype": "Q4_K"},
            {"name": "blk.0.ffn_down.weight", "shape": [512, 1024], "qtype": "Q4_K"},
            {"name": "token_embd.weight", "shape": [2048, 512], "qtype": "F16"}
        ]
    },
    {
        "tensors": [
            {"name": "blk.1.attn_v.weight", "shape": [512, 512], "qtype": "Q4_K"},
            {"name": "blk.1.attn_output.weight", "shape": [512, 512], "qtype": "Q4_K"}
        ]
    },
    {
        "tensors": [
            {"name": "blk.2.ffn_up.weight", "shape": [1024, 512], "qtype": "Q4_K"},
            {"name": "output_norm.weight", "shape": [512], "qtype": "F32"}
        ]
    },
    {
        "tensors": [
            {"name": "rope_freqs.weight", "shape": [128], "qtype": "F32"},
            {"name": "blk.3.attn_q.weight", "shape": [256, 256], "qtype": "Q4_K"}
        ]
    }
]


def tensor_bytes(tensor):
    shape = tensor["shape"]
    nelements = 1
    for dim in shape:
        nelements *= dim
    qtype = tensor["qtype"]
    if qtype == "F32":
        return nelements * 4
    elif qtype == "F16":
        return nelements * 2
    elif qtype == "Q4_K":
        return (nelements // 256) * 144
    elif qtype == "Q4_K_S":
        return (nelements // 256) * 136
    elif qtype == "Q4_K_M":
        return (nelements // 256) * 144
    return nelements * 4


def predict_output_size(inventory):
    total = 0
    for t in inventory["tensors"]:
        total += tensor_bytes(t)
    return total


def resolve_recipe(tensor_name, base_ftype):
    if "attn_q" in tensor_name or "ffn_gate" in tensor_name:
        return "Q4_K_M"
    return base_ftype


def explain_delta(tensor):
    shape = tensor["shape"]
    nelements = 1
    for dim in shape:
        nelements *= dim
    blocks = nelements // 256
    return blocks * 8
