import numpy as np

def run_sweep(teacher_logits, student_logits, labels, temperatures, alphas):
    grid = np.zeros((len(temperatures), len(alphas)))
    for i, t in enumerate(temperatures):
        for j, a in enumerate(alphas):
            p_t = np.exp((teacher_logits - np.max(teacher_logits, axis=-1, keepdims=True)) / t)
            p_t /= np.sum(p_t, axis=-1, keepdims=True)
            p_s = np.exp((student_logits - np.max(student_logits, axis=-1, keepdims=True)) / t)
            p_s /= np.sum(p_s, axis=-1, keepdims=True)
            kl = np.sum(p_t * (np.log(np.clip(p_t, 1e-12, 1.0)) - np.log(np.clip(p_s, 1e-12, 1.0))), axis=-1).mean()

            p_s_base = np.exp((student_logits - np.max(student_logits, axis=-1, keepdims=True)))
            p_s_base /= np.sum(p_s_base, axis=-1, keepdims=True)
            ce = -np.mean(np.sum(labels * np.log(np.clip(p_s_base, 1e-12, 1.0)), axis=-1))

            grid[i, j] = a * (t ** 2) * kl + (1.0 - a) * ce
    return grid
