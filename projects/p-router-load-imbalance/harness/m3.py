import sys
import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from moe.metrics import compute_imbalance_ratio, measure_distribution
        from moe.router import MoERouter
    except Exception:
        return {"aux_loss_computed": 0.0, "loss_decreases_imbalance": 0.0}

    router = MoERouter(num_experts=4, in_dim=8)
    x = ref.generate_synthetic_inputs(num_samples=100, in_dim=8, seed=123)

    probs, selected, _ = router.route(x, top_k=2)
    try:
        loss, _ = router.compute_aux_loss(probs, selected)
    except Exception:
        return {"aux_loss_computed": 0.0, "loss_decreases_imbalance": 0.0}

    counts_before = measure_distribution(selected, 4)
    imb_before = compute_imbalance_ratio(counts_before)

    aux_ok = 1.0 if isinstance(loss, float) and loss > 0 else 0.0

    for _ in range(20):
        probs, selected, _ = router.route(x, top_k=2)
        P = np.mean(probs, axis=0)
        E = 4
        N = x.shape[0]
        c = measure_distribution(selected, E)
        f = c / (N * 2)
        grad = np.dot(x.T, probs - f) * 0.01
        router.update_weights(grad, lr=0.05)

    probs_after, selected_after, _ = router.route(x, top_k=2)
    counts_after = measure_distribution(selected_after, 4)
    imb_after = compute_imbalance_ratio(counts_after)

    imbalance_decreased = 1.0 if imb_after <= imb_before else 0.0

    return {
        "aux_loss_computed": aux_ok,
        "loss_decreases_imbalance": imbalance_decreased,
    }
