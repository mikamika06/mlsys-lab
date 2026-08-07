import sys
import numpy as np


def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    import ref
    from heal.model import SimpleMLP
    from heal.pruner import Pruner
    from heal.trainer import HealerTrainer

    m = {"history_length_correct": 0.0, "loss_decreases": 0.0, "trajectory_smooth": 0.0}

    X, y = ref.get_dataset()
    model = SimpleMLP(seed=42)
    ref.pretrain_model(model, X, y, steps=150)
    pruner = Pruner(model)
    mask_mgr = pruner.prune_by_magnitude(0.5)

    trainer = HealerTrainer(model, mask_mgr, lr=0.02, max_steps=50)
    history = trainer.train(X, y, batch_size=64)

    if len(history) == 50 and len(trainer.history) == 50:
        m["history_length_correct"] = 1.0

    first_half_avg = np.mean(history[:10])
    last_half_avg = np.mean(history[-10:])
    if last_half_avg < first_half_avg:
        m["loss_decreases"] = 1.0

    if last_half_avg < first_half_avg * 0.95:
        m["trajectory_smooth"] = 1.0

    return m
