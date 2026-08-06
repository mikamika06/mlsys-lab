CONFIGS = [
    {
        "vocab_size": 32000,
        "hidden_dim": 4096,
        "layers": 32,
        "tensors": [
            {"name": "token_embd.weight", "shape": [32000, 4096], "ftype": 0},
            {"name": "blk.0.attn_q.weight", "shape": [4096, 4096], "ftype": 0},
            {"name": "output.weight", "shape": [32000, 4096], "ftype": 0}
        ]
    },
    {
        "vocab_size": 128256,
        "hidden_dim": 8192,
        "layers": 80,
        "tensors": [
            {"name": "token_embd.weight", "shape": [128256, 8192], "ftype": 2},
            {"name": "blk.0.attn_q.weight", "shape": [8192, 8192], "ftype": 2},
            {"name": "output.weight", "shape": [128256, 8192], "ftype": 2}
        ]
    },
    {
        "vocab_size": 32000,
        "hidden_dim": 2048,
        "layers": 22,
        "tensors": [
            {"name": "token_embd.weight", "shape": [32000, 2048], "ftype": 7},
            {"name": "blk.0.attn_q.weight", "shape": [2048, 2048], "ftype": 7},
            {"name": "output.weight", "shape": [32000, 2048], "ftype": 7}
        ]
    }
]


def parse_tensors(config):
    return config["tensors"]


def is_output_tensor(name):
    return "output" in name or "lm_head" in name


def tensor_bytes(shape, ftype):
    num_elements = 1
    for dim in shape:
        num_elements *= dim
    if ftype == 0:
        return num_elements * 4
    elif ftype == 2:
        return num_elements * 1
    elif ftype == 7:
        return (num_elements // 32) * 18
    return num_elements * 4


def model_total_bytes(tensors, ftype, leave_output=True):
    total = 0
    for t in tensors:
        if not leave_output and is_output_tensor(t["name"]):
            continue
        total += tensor_bytes(t["shape"], ftype)
    return total


def estimate_difference(tensors, ftype):
    with_out = model_total_bytes(tensors, ftype, leave_output=True)
    without_out = model_total_bytes(tensors, ftype, leave_output=False)
    return with_out - without_out
