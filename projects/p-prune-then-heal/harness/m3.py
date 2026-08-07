import sys
import numpy as np


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    import ref
    from heal.model import SimpleMLP
    from heal.pruner import Pruner
    from heal.trainer import HealerTrainer

    m = {"grad_masking_active": 0.0, "sparsity_strictly_preserved": 0.0, "mask_reapplied_after_step": 0.0}

    X, y = ref.get_dataset()
    model = SimpleMLP(seed=42)
    ref.pretrain_model(model, X, y, steps=150)
    pruner = Pruner(model)
    mask_mgr = pruner.prune_by_magnitude(0.5)

    initial_sparsity = mask_mgr.get_sparsity()

    test_grads = [np.ones_like(w) for w in model.weights]
    mask_mgr.mask_gradients(test_grads)
    all_masked = True
    for g, mask in zip(test_grads, mask_mgr.masks):
        if np.any(g[~mask] != 0.0):
            all_masked = False
            break
    if all_masked:
        m["grad_masking_active"] = 1.0

    trainer = HealerTrainer(model, mask_mgr, lr=0.02, max_steps=20)
    for _ in range(20):
        trainer.step(X[:64], y[:64])

    current_sparsity = mask_mgr.get_sparsity()
    if abs(current_sparsity - initial_sparsity) < 1e-6:
        m["sparsity_strictly_preserved"] = 1.0

    for w, mask in zip(model.weights, mask_mgr.masks):
        w[~mask] = 999.0
    mask_mgr.apply_mask()
    zero_restored = True
    for w, mask in zip(model.weights, mask_mgr.masks):
        if np.any(w[~mask] != 0.0):
            zero_restored = False
            break
    if zero_restored:
        m["mask_reapplied_after_step"] = 1.0

    return m
