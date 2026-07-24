import numpy as np


def _route(router_logits, num_devices, E):
    expert_id = np.argmax(router_logits, axis=1)
    experts_per_device = E // num_devices
    device_id = expert_id // experts_per_device
    return expert_id, device_id


def _single_device_reference(X, router_logits, expert_weight, num_devices):
    N, d = X.shape
    E = router_logits.shape[1]
    expert_id, device_id = _route(router_logits, num_devices, E)

    out = np.empty((N, d), dtype=np.float64)
    for i in range(N):
        out[i] = X[i] @ expert_weight[expert_id[i]]

    counts = np.bincount(device_id, minlength=num_devices).astype(np.int64)
    return out, counts


def _build_case(seed, N, d, E, num_devices):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, d)).astype(np.float64)
    router_logits = rng.standard_normal((N, E)).astype(np.float64)
    expert_weight = (rng.standard_normal((E, d, d)) * 0.5).astype(np.float64)
    return X, router_logits, expert_weight, num_devices


def grade(sol, fx) -> dict:
    worst_err = 0.0
    counts_ok = 1.0

    for seed, N, d, E, num_devices in [
        (0, 12, 4, 6, 3),
        (1, 20, 5, 8, 4),
        (2, 9, 3, 6, 2),
    ]:
        X, router_logits, expert_weight, num_devices = _build_case(seed, N, d, E, num_devices)
        ref_out, ref_counts = _single_device_reference(X, router_logits, expert_weight, num_devices)

        try:
            got_out, got_counts = sol.moe_all_to_all_dispatch(
                X.copy(), router_logits.copy(), expert_weight.copy(), num_devices
            )
            got_out = np.asarray(got_out, dtype=np.float64)
            got_counts = np.asarray(got_counts).astype(np.int64)
        except Exception:
            return {"max_abs_err": float("inf"), "counts_exact_match": 0.0}

        if got_out.shape != ref_out.shape or not np.all(np.isfinite(got_out)):
            return {"max_abs_err": float("inf"), "counts_exact_match": 0.0}

        worst_err = max(worst_err, float(np.max(np.abs(got_out - ref_out))))

        if got_counts.shape != ref_counts.shape or not np.array_equal(got_counts, ref_counts):
            counts_ok = 0.0

    return {"max_abs_err": worst_err, "counts_exact_match": counts_ok}
