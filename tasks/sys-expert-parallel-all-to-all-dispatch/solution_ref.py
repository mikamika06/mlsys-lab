import numpy as np


def moe_all_to_all_dispatch(X: np.ndarray, router_logits: np.ndarray,
                             expert_weight: np.ndarray, num_devices: int):
    """Simulate expert-parallel MoE dispatch: route -> all-to-all -> expert
    compute -> all-to-all back.

    See task.md for the exact routing/placement rule.
    """
    X = np.asarray(X, dtype=np.float64)
    router_logits = np.asarray(router_logits, dtype=np.float64)
    expert_weight = np.asarray(expert_weight, dtype=np.float64)

    N, d = X.shape
    E = router_logits.shape[1]
    experts_per_device = E // num_devices

    # top-1 routing decision, made once, locally (same on every device)
    expert_id = np.argmax(router_logits, axis=1)
    device_id = expert_id // experts_per_device

    device_counts = np.zeros(num_devices, dtype=np.int64)
    output = np.zeros((N, d), dtype=np.float64)

    for dev in range(num_devices):
        # --- all-to-all dispatch: gather every token routed to `dev` ---
        idx = np.nonzero(device_id == dev)[0]
        device_counts[dev] = idx.shape[0]
        if idx.size == 0:
            continue
        dev_tokens = X[idx]
        dev_experts = expert_id[idx]

        # --- local expert compute, resident experts only ---
        dev_out = np.empty((idx.size, d), dtype=np.float64)
        for e in np.unique(dev_experts):
            m = dev_experts == e
            dev_out[m] = dev_tokens[m] @ expert_weight[e]

        # --- all-to-all combine: scatter results back to original slots ---
        output[idx] = dev_out

    return output, device_counts
