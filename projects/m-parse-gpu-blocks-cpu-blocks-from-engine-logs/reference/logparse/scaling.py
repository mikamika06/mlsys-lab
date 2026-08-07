def verify_tp_scaling(tp1_blocks, tp2_blocks):
    ratio = tp2_blocks / max(1, tp1_blocks)
    return ratio > 1.5 and ratio <= 2.0
