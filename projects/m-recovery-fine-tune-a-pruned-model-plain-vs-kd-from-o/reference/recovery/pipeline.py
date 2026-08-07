import numpy as np


def run_recovery(model, teacher, dataset, config):
    steps = config.get("steps", 20)
    lr = config.get("lr", 0.01)
    mode = config.get("mode", "plain")
    temperature = config.get("temperature", 2.0)
    alpha = config.get("alpha", 0.5)

    history = []
    np.random.seed(42)

    for step in range(steps):
        for x, y in dataset:
            s_logits = model(x)
            if mode == "kd" and teacher is not None:
                t_logits = teacher(x)
                soft_t = np.exp(t_logits / temperature) / np.sum(np.exp(t_logits / temperature), axis=-1, keepdims=True)
                soft_s = np.exp(s_logits / temperature) / np.sum(np.exp(s_logits / temperature), axis=-1, keepdims=True)
                kd_loss = np.sum(-soft_t * np.log(soft_s + 1e-8)) * (temperature ** 2)
                ce_loss = np.mean((s_logits - y) ** 2)
                loss = alpha * kd_loss + (1 - alpha) * ce_loss
                grad = (s_logits - y) * 0.05 + alpha * (soft_s - soft_t) * 0.01
            else:
                loss = np.mean((s_logits - y) ** 2)
                grad = (s_logits - y) * 0.05

            model.weights -= lr * np.mean(grad)

        acc = float(np.mean(np.abs(model.weights) > 0.1))
        history.append(acc)

    return history


def evaluate_accuracy(model, dataset):
    preds = [np.mean(model(x)) for x, y in dataset]
    targets = [np.mean(y) for x, y in dataset]
    acc = 1.0 - float(np.mean(np.abs(np.array(preds) - np.array(targets))))
    return max(0.0, min(1.0, acc))
