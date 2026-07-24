CONFIGS = [
    (64, 128, True),
    (64, 128, False),
    (10, 40, True),
    (10, 40, False),
    (32, 256, True),
    (17, 60, False),
]


def _oracle(d_out, d_in, use_int4):
    n_groups = d_in // 4
    kept = d_out * n_groups * 2  # 2 nonzeros per group of 4
    meta_bits = kept * 2  # 2-bit position code per kept value
    meta_bytes = -(-meta_bits // 8)  # ceil

    if use_int4:
        value_bytes = -(-kept // 2)  # 2 nibbles per byte
        scale_bytes = d_out * 2  # one fp16 scale per row
    else:
        value_bytes = kept * 2  # fp16 values, no scale needed
        scale_bytes = 0

    total = value_bytes + meta_bytes + scale_bytes
    dense = d_out * d_in * 2  # dense fp16
    return total, dense / total


def grade(sol, fx) -> dict:
    """
    Computes the served byte count and size_ratio (vs dense fp16) of a
    2:4-structured-sparse weight, optionally with int4-packed kept
    values, with a formula oracle over a fixed set of (d_out, d_in,
    use_int4) configurations (d_in always divisible by 4). Compares the
    submission's total bytes (exact) and size_ratio (relative error) to
    the oracle's, worst case across configurations.
    """
    bytes_ok = 1.0
    ratio_rel_worst = 0.0
    for d_out, d_in, use_int4 in CONFIGS:
        total_exp, ratio_exp = _oracle(d_out, d_in, use_int4)

        try:
            total_got, ratio_got = sol.sparse_2_4_footprint(d_out, d_in, use_int4)
            total_got = int(total_got)
            ratio_got = float(ratio_got)
        except Exception:
            return {"bytes_exact_match": 0.0, "ratio_rel_err": float("inf")}

        if total_got != total_exp:
            bytes_ok = 0.0
        ratio_rel = abs(ratio_got - ratio_exp) / (abs(ratio_exp) + 1e-12)
        ratio_rel_worst = max(ratio_rel_worst, ratio_rel)

    return {"bytes_exact_match": bytes_ok, "ratio_rel_err": ratio_rel_worst}
