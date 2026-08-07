import sys
import numpy as np
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    try:
        from moe.router import MoERouter
    except Exception:
        return {"quality_preserved": 0.0}

    router = MoERouter(num_experts=8, in_dim=16)
    x = ref.generate_synthetic_inputs(num_samples=200, in_dim=16, seed=999)

    probs_initial, _, weights_initial = router.route(x, top_k=2)
    quality_score_initial = float(
        np.mean(np.max(probs_initial, axis=-1))
        + np.mean(weights_initial)
    )

    for _ in range(10):
        probs, selected, _ = router.route(x, top_k=2)
        grad = np.random.randn(*router.W.shape) * 0.001
        router.update_weights(grad, lr=0.01)

    probs_final, _, weights_final = router.route(x, top_k=2)
    quality_score_final = float(
        np.mean(np.max(probs_final, axis=-1)) + np.mean(weights_final)
    )

    preserved = (
        1.0 if quality_score_final >= 0.85 * quality_score_initial else 0.0
    )
    return {"quality_preserved": preserved}
