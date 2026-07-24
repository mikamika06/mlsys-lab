OP_KIND = {
    "add": "ew", "mul": "ew", "sub": "ew", "div": "ew",
    "relu": "ew", "gelu": "ew", "sigmoid": "ew", "silu": "ew",
    "sum": "reduce", "mean": "reduce", "max": "reduce", "softmax": "reduce", "layernorm": "reduce",
    "matmul": "gemm", "conv2d": "gemm", "linear": "gemm",
    "reshape": "reshape", "transpose": "reshape", "view": "reshape", "permute": "reshape",
}

FUSE_RULE = {
    ("ew", "ew"): True,
    ("ew", "reduce"): True,
    ("reduce", "ew"): True,
    ("reduce", "reduce"): False,
    ("ew", "gemm"): False,
    ("gemm", "ew"): True,
    ("gemm", "gemm"): False,
    ("gemm", "reduce"): False,
    ("reduce", "gemm"): False,
}


def classify_fusable(op_pairs):
    """
    For each (producer_op_name, consumer_op_name) pair, classify whether
    the producer can be fused directly into the consumer's kernel:

    - Map each op name to its kind via OP_KIND (ew / reduce / gemm / reshape).
    - reshape/transpose/view/permute is a free metadata op: fuses with
      anything, in either position.
    - Otherwise look up (producer_kind, consumer_kind) in FUSE_RULE.

    Returns a list of bool, one per input pair, in order.
    """
    out = []
    for a, b in op_pairs:
        ka, kb = OP_KIND[a], OP_KIND[b]
        if ka == "reshape" or kb == "reshape":
            out.append(True)
        else:
            out.append(FUSE_RULE[(ka, kb)])
    return out
