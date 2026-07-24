import numpy as np

def kd_loss(teacher_logits: np.ndarray, student_logits: np.ndarray, labels: np.ndarray, alpha: float=0.5, temperature: float=1.0) -> float:
    raise NotImplementedError('your code here')
