import numpy as np


def softmax(logits):
    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp / np.sum(exp, axis=-1, keepdims=True)


def ref_plain_loss(weights, X, y):
    logits = np.dot(X, weights)
    probs = softmax(logits)
    m = y.shape[0]
    log_probs = -np.log(probs[np.arange(m), y] + 1e-12)
    return float(np.mean(log_probs))


def ref_kd_loss(student_weights, teacher_weights, X, temperature=2.0, alpha=0.5):
    student_logits = np.dot(X, student_weights) / temperature
    teacher_logits = np.dot(X, teacher_weights) / temperature
    student_probs = softmax(student_logits)
    teacher_probs = softmax(teacher_logits)
    kl_div = np.sum(teacher_probs * (np.log(teacher_probs + 1e-12) - np.log(student_probs + 1e-12)), axis=-1)
    return float(np.mean(kl_div) * (temperature ** 2) * alpha)


def ref_evaluate_accuracy(weights, X, y):
    logits = np.dot(X, weights)
    preds = np.argmax(logits, axis=-1)
    return float(np.mean(preds == y))


def ref_steps_to_90_recovery(accuracies, baseline_acc, pruned_acc):
    threshold = pruned_acc + 0.9 * (baseline_acc - pruned_acc)
    for idx, acc in enumerate(accuracies):
        if acc >= threshold:
            return int(idx)
    return int(len(accuracies))


def generate_fixture():
    np.random.seed(123)
    X = np.random.randn(40, 8)
    y = np.random.randint(0, 2, size=(40,))
    teacher = np.random.randn(8, 2)
    student = teacher * (np.random.rand(8, 2) > 0.4)
    accuracies = [0.5, 0.55, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    return X, y, teacher, student, accuracies, 0.95, 0.5
