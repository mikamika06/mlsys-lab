import numpy as np

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


def _fusable(producer_kind, consumer_kind):
    if producer_kind == "reshape" or consumer_kind == "reshape":
        return True  # a reshape/transpose/view is a free metadata op
    return FUSE_RULE[(producer_kind, consumer_kind)]


def _oracle(pairs):
    return [_fusable(OP_KIND[a], OP_KIND[b]) for a, b in pairs]


def grade(sol, fx) -> dict:
    """
    Builds a seeded random list of (producer_op, consumer_op) name pairs
    and computes the ground-truth fusability label for each, using the
    op-kind classification and fusion rule table above (elementwise-
    elementwise fuses; elementwise<->reduction fuses; reshape/transpose/
    view fuses with anything (free, no compute); reduction-reduction,
    elementwise->gemm, and any pair involving two gemms does not).
    Compares the submission's labels to the oracle's, exactly.
    """
    rng = np.random.default_rng(0)
    names = list(OP_KIND.keys())
    n = 40
    idx_a = rng.integers(0, len(names), n)
    idx_b = rng.integers(0, len(names), n)
    pairs = [(names[i], names[j]) for i, j in zip(idx_a, idx_b)]

    labels_exp = _oracle(pairs)

    try:
        labels_got = sol.classify_fusable(list(pairs))
        labels_got = [bool(x) for x in labels_got]
    except Exception:
        return {"exact_match": 0.0}

    if len(labels_got) != len(labels_exp):
        return {"exact_match": 0.0}

    match = 1.0 if labels_got == labels_exp else 0.0
    return {"exact_match": match}
