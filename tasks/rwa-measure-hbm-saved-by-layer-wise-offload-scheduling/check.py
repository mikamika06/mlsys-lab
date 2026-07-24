def _oracle(n_layers, active_layers, per_layer_kv_bytes):
    full_bytes = n_layers * per_layer_kv_bytes
    resident_bytes = active_layers * per_layer_kv_bytes
    ratio = resident_bytes / full_bytes if full_bytes else 0.0
    return {
        "peak_resident_bytes": resident_bytes,
        "resident_ratio": ratio,
    }


def grade(sol, fx) -> dict:
    cases = [
        (32, 4, 1024 * 1024),
        (80, 8, 4096),
        (12, 3, 777),
        (48, 16, 12345),
        (7, 1, 999999),
    ]

    ok = 1.0
    for n_layers, active_layers, per_layer_kv_bytes in cases:
        expected = _oracle(n_layers, active_layers, per_layer_kv_bytes)
        try:
            got = sol.measure_hbm_saved(
                n_layers,
                active_layers,
                per_layer_kv_bytes,
            )
        except Exception:
            ok = 0.0
            break

        if not isinstance(got, dict):
            ok = 0.0
            break

        if set(got.keys()) != {
            "peak_resident_bytes",
            "resident_ratio",
        }:
            ok = 0.0
            break

        if (
            got["peak_resident_bytes"] != expected["peak_resident_bytes"]
            or got["resident_ratio"] != expected["resident_ratio"]
        ):
            ok = 0.0
            break

    return {"size_ratio": ok}
