import numpy as np


def hidden_distillation_loss(teacher, student, projection):
    teacher = np.asarray(teacher, dtype=np.float64)
    student = np.asarray(student, dtype=np.float64)
    projection = np.asarray(projection, dtype=np.float64)
    
    m = student.shape[0]
    n = projection.shape[1]
    
    projected = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            s = 0.0
            for k in range(student.shape[1]):
                s += student[i, k] * projection[k, j]
            projected[i, j] = s
            
    diff = np.zeros(teacher.shape, dtype=np.float64)
    for i in range(teacher.shape[0]):
        for j in range(teacher.shape[1]):
            diff[i, j] = teacher[i, j] - projected[i, j]
            
    total_elements = teacher.shape[0] * teacher.shape[1]
    acc = 0.0
    for i in range(teacher.shape[0]):
        for j in range(teacher.shape[1]):
            val = diff[i, j]
            acc += val * val
            
    return float(acc / total_elements)
