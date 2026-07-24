def _oracle(d, r, num_layers, N, base_params_per_layer, dtype_bytes):
    """Recompute both memory budgets from scratch."""
    base_total = base_params_per_layer * num_layers
    adapter_params = N * 2 * d * r * num_layers
    adapter_bytes = (base_total + adapter_params) * dtype_bytes
    full_bytes = base_total * N * dtype_bytes
    return (adapter_bytes, full_bytes)

def grade(sol, fx) -> dict:
    cases = [
        # fixture / example case: 1 GiB vs 6 GiB
        dict(d=4096, r=64, num_layers=32, N=8,
             base_params_per_layer=12_582_912, dtype_bytes=2),
        # small case
        dict(d=128, r=8, num_layers=4, N=2,
             base_params_per_layer=500_000, dtype_bytes=4),
        # edge: full-rank adapter (r == d)
        dict(d=256, r=256, num_layers=2, N=1,
             base_params_per_layer=2_000_000, dtype_bytes=2),
        # degenerate: N == 1 (adapters barely win)
        dict(d=512, r=16, num_layers=12, N=1,
             base_params_per_layer=1_000_000, dtype_bytes=2),
        # large-N stress
        dict(d=2048, r=32, num_layers=24, N=16,
             base_params_per_layer=5_000_000, dtype_bytes=2),
    ]
    ok = 1.0
    for c in cases:
        expected = _oracle(**c)
        try:
            got = sol.memory_comparison(**c)
            got = (int(got[0]), int(got[1]))
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
