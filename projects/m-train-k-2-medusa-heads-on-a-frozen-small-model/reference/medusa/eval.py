import numpy as np


def evaluate_head_accuracy(medusa_heads, hidden_states, targets):
    logits = medusa_heads.forward(hidden_states)
    accuracies = []
    N, T, _ = hidden_states.shape

    for k in range(2):
        shift = k + 1
        if T <= shift:
            accuracies.append(0.0)
            continue
        pred = np.argmax(logits[k][:, : T - shift, :], axis=-1)
        tgt = targets[:, shift:T]
        acc = float(np.mean(pred == tgt))
        accuracies.append(acc)

    return accuracies


def compare_vicuna_benchmark(head2_acc, target_acc=0.60, tol=0.15):
    diff = abs(head2_acc - target_acc)
    within_bounds = bool(diff <= tol)
    rel_err = float(diff / target_acc)
    return {
        "within_bounds": within_bounds,
        "rel_err": rel_err,
        "head2_acc": float(head2_acc),
        "target_acc": float(target_acc),
    }
