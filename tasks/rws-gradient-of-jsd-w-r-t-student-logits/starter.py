import numpy as np


def jsd_grad_wrt_student_logits(
    teacher_logits: np.ndarray,
    student_logits: np.ndarray,
    beta: float,
) -> np.ndarray:
    """p = softmax(teacher_logits) (fixed), q = softmax(student_logits).
    m = beta*p + (1-beta)*q. g_j = (1-beta)*log(q_j/m_j) = dJSD/dq_j.
    Return the softmax-Jacobian-vector product q * (g - sum(q*g)), the
    gradient of JSD_beta(p, q) w.r.t. student_logits."""
    raise NotImplementedError('your code here')
