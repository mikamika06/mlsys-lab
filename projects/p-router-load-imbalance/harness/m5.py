import sys
import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from moe.metrics import compute_imbalance_ratio, measure_distribution
        from moe.router import MoERouter
    except Exception:
        return {"imbalance_ratio_low": 0.0}

    router = MoERouter(num_experts=8, in_dim=16)
    x = ref.generate_synthetic_inputs(num_samples=400, in_dim=16, seed=2024)

    for _ in range(30):
        probs, selected, _ = router.route(x, top_k=2)
        c = measure_distribution(selected, 8)
        f = c / (x.shape[0] * 2)
        grad = np.dot(x.T, probs - f) * 0.005
        router.update_weights(grad, lr=0.02)

    _, selected_final, _ = router.route(x, top_k=2)
    final_counts = measure_distribution(selected_final, 8)
    imb_ratio = compute_imbalance_ratio(final_counts)

    return {"imbalance_ratio_low": 1.0 if imb_ratio < 2.0 else 0.0}
