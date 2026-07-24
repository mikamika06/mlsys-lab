import numpy as np

def kl_divergences(teacher_logits: np.ndarray, student_logits: np.ndarray, temperature: float):
    """
    Compute forward and reverse KL divergences between teacher and student logits.
    Parameters
    ----------
    teacher_logits : np.ndarray
        1-D array of teacher logits.
    student_logits : np.ndarray
        1-D array of student logits.
    temperature : float
        Positive temperature for softmax scaling.

    Returns
    -------
    Tuple[float, float]
        (forward_kl, reverse_kl)
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    # Softmax with temperature
    t = teacher_logits / temperature
    s = student_logits / temperature
    t_max = np.max(t)
    s_max = np.max(s)
    exp_t = np.exp(t - t_max)
    exp_s = np.exp(s - s_max)
    p = exp_t / np.sum(exp_t)
    q = exp_s / np.sum(exp_s)
    eps = 1e-12
    forward_kl = float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))
    reverse_kl = float(np.sum(q * (np.log(q + eps) - np.log(p + eps))))
    return forward_kl, reverse_kl
