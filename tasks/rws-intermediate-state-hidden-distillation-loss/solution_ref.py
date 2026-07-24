import numpy as np


def hidden_distillation_loss(teacher, student, projection):
    teacher = np.asarray(teacher, dtype=np.float64)
    student = np.asarray(student, dtype=np.float64)
    projection = np.asarray(projection, dtype=np.float64)
    diff = teacher - student @ projection
    return float(np.mean(diff ** 2))
