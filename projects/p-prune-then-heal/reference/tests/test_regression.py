import sys
import numpy as np

sys.path.insert(0, ".")
import ref
from heal.model import SimpleMLP
from heal.pruner import Pruner
from heal.trainer import HealerTrainer


def test_gradients_are_masked():
    X, y = ref.get_dataset()
    model = SimpleMLP(seed=42)
    pruner = Pruner(model)
    mask_mgr = pruner.prune_by_magnitude(0.5)

    grads = [np.ones_like(w) for w in model.weights]
    mask_mgr.mask_gradients(grads)

    for g, mask in zip(grads, mask_mgr.masks):
        assert np.all(g[~mask] == 0.0), "Gradients for pruned weights were not zeroed"


def test_step_budget_exceeded_raises():
    X, y = ref.get_dataset()
    model = SimpleMLP(seed=42)
    ref.pretrain_model(model, X, y, steps=50)
    pruner = Pruner(model)
    mask_mgr = pruner.prune_by_magnitude(0.5)

    trainer = HealerTrainer(model, mask_mgr, lr=0.02, max_steps=3)
    for _ in range(3):
        trainer.step(X[:32], y[:32])

    raised = False
    try:
        trainer.step(X[:32], y[:32])
    except RuntimeError:
        raised = True

    assert raised, "Expected RuntimeError when stepping past max_steps"
