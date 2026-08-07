def hidden_distillation_loss(teacher, student, projection):
    m = len(student)
    n = len(projection[0])
    k_dim = len(student[0])

    projected = [[0.0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = 0.0
            for k in range(k_dim):
                s += student[i][k] * projection[k][j]
            projected[i][j] = s

    n_teacher = len(teacher)
    d_teacher = len(teacher[0])

    total_elements = n_teacher * d_teacher
    acc = 0.0
    for i in range(n_teacher):
        for j in range(d_teacher):
            diff = teacher[i][j] - projected[i][j]
            acc += diff * diff

    return float(acc / total_elements)
