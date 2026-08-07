import sys


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    import ref
    from heal.model import SimpleMLP
    from heal.pruner import Pruner

    m = {"mask_created": 0.0, "sparsity_target_met": 0.0, "baseline_captured": 0.0}

    X, y = ref.get_dataset()
    model = SimpleMLP(seed=42)
    ref.pretrain_model(model, X, y, steps=150)

    pruner = Pruner(model)
    mask_mgr = pruner.prune_by_magnitude(0.5)

    if mask_mgr is not None and hasattr(mask_mgr, "masks") and len(mask_mgr.masks) == 2:
        m["mask_created"] = 1.0

    if mask_mgr is not None:
        sparsity = mask_mgr.get_sparsity()
        if abs(sparsity - 0.5) < 0.05:
            m["sparsity_target_met"] = 1.0

    stats = pruner.get_baseline_stats(X, y)
    if isinstance(stats, dict) and "acc" in stats and "loss" in stats and "sparsity" in stats:
        if stats["acc"] > 0.0 and stats["loss"] > 0.0 and abs(stats["sparsity"] - 0.5) < 0.05:
            m["baseline_captured"] = 1.0

    return m
