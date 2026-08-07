import numpy as np

def distill_logits(teacher_logits, student_logits, T=2.0):
    soft_t = np.exp(teacher_logits / T) / np.sum(np.exp(teacher_logits / T), axis=-1, keepdims=True)
    soft_s = np.exp(student_logits / T) / np.sum(np.exp(student_logits / T), axis=-1, keepdims=True)
    return np.mean(np.sum(-soft_t * np.log(soft_s + 1e-8), axis=-1))
