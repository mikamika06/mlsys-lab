import numpy as np

def check_pareto(teacher_acc, student_acc, teacher_size, student_size):
    size_ratio = student_size / teacher_size
    acc_drop = teacher_acc - student_acc
    return size_ratio <= 0.51 and acc_drop <= 0.02
