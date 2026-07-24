import numpy as np

def uld_gradient(student_logits, teacher_logits):
    """Gradient of ULD loss w.r.t. student_logits."""
    s = np.asarray(student_logits).ravel()
    t = np.asarray(teacher_logits).ravel()
    diff = np.sort(s) - np.sort(t)
    rank = np.argsort(s).argsort()
    return 2.0 * diff[rank]
