import numpy as np
from heal.model import SimpleMLP
from heal.pruner import Pruner
from heal.trainer import HealerTrainer


def get_dataset(n_samples=500, input_dim=20, num_classes=5, seed=42):
    rng = np.random.RandomState(seed)
    W_true = rng.randn(input_dim, num_classes)
    X = rng.randn(n_samples, input_dim)
    logits = X @ W_true + rng.randn(n_samples, num_classes) * 0.1
    y = np.argmax(logits, axis=1)
    return X, y


def pretrain_model(model, X, y, steps=150, lr=0.05, seed=42):
    rng = np.random.RandomState(seed)
    n_samples = X.shape[0]
    for _ in range(steps):
        indices = rng.choice(n_samples, size=64, replace=False)
        _, w_grads, b_grads = model.forward_backward(X[indices], y[indices])
        model.apply_gradients(w_grads, b_grads, lr)


def get_reference_pruned_model(seed=42):
    X, y = get_dataset(seed=seed)
    model = SimpleMLP(seed=seed)
    pretrain_model(model, X, y, steps=150)
    pruner = Pruner(model)
    mask_mgr = pruner.prune_by_magnitude(0.5)
    return model, mask_mgr, X, y


def get_healed_model(seed=42):
    model, mask_mgr, X, y = get_reference_pruned_model(seed=seed)
    trainer = HealerTrainer(model, mask_mgr, lr=0.03, max_steps=100)
    trainer.train(X, y, batch_size=64)
    return model, trainer
