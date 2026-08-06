import numpy as np


def measure_retention(student_logits, teacher_logits, targets):
    student_preds = np.argmax(student_logits, axis=-1)
    teacher_preds = np.argmax(teacher_logits, axis=-1)
    student_acc = np.mean(student_preds == targets)
    teacher_acc = np.mean(teacher_preds == targets)
    if teacher_acc == 0:
        return 0.0
    return float(student_acc / teacher_acc)
