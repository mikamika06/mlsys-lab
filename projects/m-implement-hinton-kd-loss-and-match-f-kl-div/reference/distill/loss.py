import torch
import torch.nn.functional as F


def hinton_kd_loss(student_logits, teacher_logits, temperature=1.0, alpha=0.5, labels=None):
    s_log_soft = F.log_softmax(student_logits / temperature, dim=-1)
    t_soft = F.softmax(teacher_logits / temperature, dim=-1)
    kd_loss = F.kl_div(s_log_soft, t_soft, reduction='batchmean') * (temperature ** 2)

    if labels is None or alpha == 0.0:
        return kd_loss

    ce_loss = F.cross_entropy(student_logits, labels)
    return alpha * ce_loss + (1.0 - alpha) * kd_loss
