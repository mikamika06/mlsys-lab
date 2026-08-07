import sys


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    import ref
    from heal.model import SimpleMLP
    from heal.pruner import Pruner
    from heal.trainer import HealerTrainer

    m = {"healing_completed": 0.0, "recovered_accuracy_ratio": 0.0, "final_acc_above_threshold": 0.0}

    X, y = ref.get_dataset()
    model = SimpleMLP(seed=42)
    ref.pretrain_model(model, X, y, steps=150)

    acc_orig, _ = model.evaluate(X, y)

    pruner = Pruner(model)
    mask_mgr = pruner.prune_by_magnitude(0.5)
    acc_pruned, _ = model.evaluate(X, y)

    trainer = HealerTrainer(model, mask_mgr, lr=0.03, max_steps=100)
    trainer.train(X, y, batch_size=64)

    if trainer.step_count == 100:
        m["healing_completed"] = 1.0

    acc_healed, _ = model.evaluate(X, y)

    drop = acc_orig - acc_pruned
    recovered = acc_healed - acc_pruned

    if drop > 1e-6:
        ratio = float(recovered / drop)
    else:
        ratio = 1.0

    m["recovered_accuracy_ratio"] = max(0.0, min(1.0, ratio))
    m["final_acc_above_threshold"] = float(acc_healed)

    return m
