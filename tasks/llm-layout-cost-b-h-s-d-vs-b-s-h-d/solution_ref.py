import numpy as np

def layout_cost(shape):
    """
    Compute cache‑line access counts for BHSD and BSHD layouts.
    """
    B, H, S, d = shape
    # strides in C‑order
    strides_bh = [H * S * d, S * d, d, 1]
    strides_sh = [S * H * d, H * d, d, 1]

    line_size_elems = 8  # 64 bytes / 8 bytes per float64

    def cache_lines(traversal_shape, strides):
        seen = set()
        for b in range(traversal_shape[0]):
            for i1 in range(traversal_shape[1]):
                for i2 in range(traversal_shape[2]):
                    for i3 in range(traversal_shape[3]):
                        offset = (
                            b * strides[0]
                            + i1 * strides[1]
                            + i2 * strides[2]
                            + i3 * strides[3]
                        )
                        seen.add(offset // line_size_elems)
        return len(seen)

    cost_bh = cache_lines((B, H, S, d), strides_bh)
    cost_sh = cache_lines((B, S, H, d), strides_sh)
    return cost_bh, cost_sh
