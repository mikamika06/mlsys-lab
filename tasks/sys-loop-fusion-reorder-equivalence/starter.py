def fused_tile_pipeline(A, B, C, tile_size):
    """
    Compute D = (A + B) * C; E = relu(D) - A; F = sum(E), processing the
    1-D arrays A, B, C tile-by-tile: for each contiguous tile of up to
    `tile_size` elements (the last tile may be shorter), apply add, relu,
    and subtract fused in one pass over that tile, and accumulate that
    tile's partial sum into the running total F.

    Return (E, F): E is the full array of per-element results (same shape
    as A), F is the scalar sum(E) as a python float.
    """
    raise NotImplementedError('your code here')
