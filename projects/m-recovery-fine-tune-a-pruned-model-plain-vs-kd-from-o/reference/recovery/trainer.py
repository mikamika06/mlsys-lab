import numpy as np


def softmax(logits):
    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp / np.sum(exp, axis=-1, keepdims=True)


def compute_plain_loss(weights, X, y):
    logits = np.dot(X, weights)
    probs = softmax(logits)
    m = y.shape[0]
    log_probs = -np.log(probs[np.arange(m), y] + 1e-12)
    return float(np.mean(log_probs))


def compute_kd_loss(student_weights, teacher_weights, X, temperature=2.0, alpha=0.5):
    student_logits = np.dot(X, student_weights) / temperature
    teacher_logits = np.dot(X, teacher_weights) / temperature
    student_probs = softmax(student_logits)
    teacher_probs = softmax(teacher_logits)
    kl_div = np.sum(teacher_probs * (np.log(teacher_probs + 1e-12) - np.log(student_probs + 1e-12)), axis=-1)
    return float(np.mean(kl_div) * (temperature ** 2) * alpha)
