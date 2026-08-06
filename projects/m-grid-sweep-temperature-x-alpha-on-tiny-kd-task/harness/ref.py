import numpy as np

def softmax(x, t=1.0):
    e = np.exp((x - np.max(x, axis=-1, keepdims=True)) / t)
    return e / np.sum(e, axis=-1, keepdims=True)

def cross_entropy(y_true, probs):
    return -np.mean(np.sum(y_true * np.log(np.clip(probs, 1e-12, 1.0)), axis=-1))

def kl_div(p, q):
    return np.sum(p * (np.log(np.clip(p, 1e-12, 1.0)) - np.log(np.clip(q, 1e-12, 1.0))), axis=-1).mean()

def compute_kd_loss(teacher_logits, student_logits, labels, T, alpha):
    p_t = softmax(teacher_logits, T)
    p_s = softmax(student_logits, T)
    kl = kl_div(p_t, p_s)
    probs_s = softmax(student_logits, 1.0)
    ce = cross_entropy(labels, probs_s)
    return alpha * (T ** 2) * kl + (1.0 - alpha) * ce

def run_sweep(teacher_logits, student_logits, labels, temperatures, alphas):
    grid = np.zeros((len(temperatures), len(alphas)))
    for i, t in enumerate(temperatures):
        for j, a in enumerate(alphas):
            grid[i, j] = compute_kd_loss(teacher_logits, student_logits, labels, t, a)
    return grid

def verify_gradient_scaling(teacher_logits, student_logits, T):
    eps = 1e-5
    s = student_logits.copy()
    s[0, 0] += eps
    loss_plus = compute_kd_loss(teacher_logits, s, np.array([[1.0, 0.0, 0.0]]), T, 1.0)
    s[0, 0] -= 2 * eps
    loss_minus = compute_kd_loss(teacher_logits, s, np.array([[1.0, 0.0, 0.0]]), T, 1.0)
    num_grad = (loss_plus - loss_minus) / (2 * eps)
    return float(np.abs(num_grad))

def effective_temperature(teacher_logits, noise_std, T):
    rng = np.random.default_rng(42)
    noisy = teacher_logits + rng.normal(0, noise_std, size=teacher_logits.shape)
    p_orig = softmax(teacher_logits, T)
    p_noisy = softmax(noisy, T)
    diff = np.mean(np.abs(p_orig - p_noisy))
    return float(diff * T)
