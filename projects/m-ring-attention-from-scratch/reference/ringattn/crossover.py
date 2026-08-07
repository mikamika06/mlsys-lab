def compute_crossover(seq_len, hidden_dim, world_size):
    """Compute communication volume for ring vs ulysses."""
    ring_vol = 2 * (world_size - 1) * seq_len * hidden_dim
    ulysses_vol = 2 * (world_size - 1) * seq_len * hidden_dim / (world_size * world_size)
    return {
        "ring": float(ring_vol),
        "ulysses": float(ulysses_vol),
        "better": "ulysses" if ulysses_vol < ring_vol else "ring"
    }
