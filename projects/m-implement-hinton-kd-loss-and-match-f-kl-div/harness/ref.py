import torch
import torch.nn.functional as F


def get_test_cases():
    torch.manual_seed(1337)
    cases = []
    for _ in range(5):
        bs = torch.randint(2, 8, (1,)).item()
        vocab = torch.randint(5, 20, (1,)).item()
        s = torch.randn(bs, vocab)
        t = torch.randn(bs, vocab)
        temp = float(torch.randint(1, 5, (1,)).item())
        alpha = float(torch.rand(1).item())
        labels = torch.randint(0, vocab, (bs,))
        cases.append((s, t, temp, alpha, labels))
    return cases


def ref_hinton_kd_loss(student_logits, teacher_logits, temperature=1.0, alpha=0.5, labels=None):
    s_log_soft = F.log_softmax(student_logits / temperature, dim=-1)
    t_soft = F.softmax(teacher_logits / temperature, dim=-1)
    kd_loss = F.kl_div(s_log_soft, t_soft, reduction='batchmean') * (temperature ** 2)
    if labels is None or alpha == 0.0:
        return kd_loss
    ce_loss = F.cross_entropy(student_logits, labels)
    return alpha * ce_loss + (1.0 - alpha) * kd_loss


def ref_softmax_entropy_curve(logits, temperatures):
    entropies = []
    for t in temperatures:
        probs = F.softmax(logits / t, dim=-1)
        log_probs = F.log_softmax(logits / t, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1).mean().item()
        entropies.append(entropy)
    return entropies
