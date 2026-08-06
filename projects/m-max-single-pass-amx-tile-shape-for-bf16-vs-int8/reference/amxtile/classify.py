from amxtile.shape import max_tile_shape


def classify_tileability(M, N, K, dtype):
    """Classify matrix block tileability."""
    rows, cols = max_tile_shape(dtype)
    return {
        "single_pass": bool(M <= rows and N <= cols),
        "max_rows": rows,
        "max_cols": cols
    }
