import numpy as np

def distill_hidden(teacher_hidden, student_hidden):
    return np.mean((teacher_hidden - student_hidden) ** 2)
