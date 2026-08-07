import sys


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    import ref
    from heal.model import SimpleMLP
    from heal.pruner import Pruner
    from heal.trainer import HealerTrainer

    m = {"trainer_initialized": 0.0, "budget_configured": 0.0, "budget_enforced": 0.0}

    X, y = ref.get_dataset()
    model = SimpleMLP(seed=42)
    ref.pretrain_model(model, X, y, steps=150)
    pruner = Pruner(model)
    mask_mgr = pruner.prune_by_magnitude(0.5)

    trainer = HealerTrainer(model, mask_mgr, lr=0.02, max_steps=10)
    if trainer.step_count == 0:
        m["trainer_initialized"] = 1.0

    if trainer.max_steps == 10:
        m["budget_configured"] = 1.0

    for _ in range(10):
        trainer.step(X[:32], y[:32])

    try:
        trainer.step(X[:32], y[:32])
        m["budget_enforced"] = 0.0
    except RuntimeError:
        m["budget_enforced"] = 1.0
    except Exception:
        m["budget_enforced"] = 0.0

    return m
