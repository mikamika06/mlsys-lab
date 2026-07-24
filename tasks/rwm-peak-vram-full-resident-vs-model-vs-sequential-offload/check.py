def grade(sol, fx) -> dict:
    """
    Compute the peak VRAM values using the same formulas as in the problem statement
    and compare them to the candidate's output.  The comparison is exact; any
    deviation causes the gate to fail.
    """
    cases = [
        (10, 4_096, 8),
        (5, 2_048, 16),
        (20, 8_192, 4)
    ]
    ok = 1.0
    for num_layers, layer_size, batch_size in cases:
        try:
            got = sol.compute_peak_vram(num_layers, layer_size, batch_size)
            if not isinstance(got, dict):
                ok = 0.0
                break
            W = num_layers * layer_size
            A = layer_size * batch_size
            M = 1024**2
            ref = {
                "full_resident": (W + A) / M,
                "model_offload": max(W, A) / M,
                "sequential_offload": max(layer_size, A) / M
            }
            # Exact match required; use a tiny tolerance to guard against
            # floating‑point representation quirks.
            if not all(abs(got[k] - ref[k]) < 1e-12 for k in ref):
                ok = 0.0
                break
        except Exception:
            ok = 0.0
            break
    return {"exact_match": ok}
