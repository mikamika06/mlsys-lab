def sparse_2_4_footprint(d_out: int, d_in: int, use_int4: bool):
    """
    Served byte count of a (d_out, d_in) weight stored in the 2:4
    structured-sparse serving format (every contiguous group of 4 columns
    keeps exactly 2 nonzeros), optionally with the kept values int4-packed:

    - kept values: d_out * (d_in // 4) * 2.
    - metadata: 2 bits per kept value (its position, 0-3, within its
      group of 4), packed to whole bytes: ceil(kept * 2 / 8).
    - values, if use_int4: ceil(kept / 2) bytes (2 nibbles/byte) plus one
      fp16 scale per row (d_out * 2 bytes).
      Otherwise: kept * 2 bytes (fp16 values), no scale.

    total = value_bytes + metadata_bytes + scale_bytes.
    size_ratio = (d_out * d_in * 2) / total   -- vs. dense fp16.

    Returns (total_bytes, size_ratio).
    """
    n_groups = d_in // 4
    kept = d_out * n_groups * 2

    meta_bits = kept * 2
    meta_bytes = -(-meta_bits // 8)

    if use_int4:
        value_bytes = -(-kept // 2)
        scale_bytes = d_out * 2
    else:
        value_bytes = kept * 2
        scale_bytes = 0

    total = value_bytes + meta_bytes + scale_bytes
    dense = d_out * d_in * 2
    return total, dense / total
