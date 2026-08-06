def verify_gradient_factor(student_logits, teacher_logits, T, alpha):
    raise NotImplementedError


def compute_effective_temperature_shift(teacher_logits, T, noise_std):
    raise NotImplementedError
