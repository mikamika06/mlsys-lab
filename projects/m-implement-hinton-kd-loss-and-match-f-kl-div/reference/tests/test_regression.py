import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from distill.loss import hinton_kd_loss


def test_hinton_kd_t2_scaling():
    torch.manual_seed(42)
    s_logits = torch.randn(4, 10, requires_grad=True)
    t_logits = torch.randn(4, 10)
    T = 3.5

    loss = hinton_kd_loss(s_logits, t_logits, temperature=T, alpha=0.0)
    loss.backward()

    grad_with_t2 = s_logits.grad.clone()
    s_logits.grad = None

    s_log_soft = F.log_softmax(s_logits / T, dim=-1)
    t_soft = F.softmax(t_logits / T, dim=-1)
    raw_kl = F.kl_div(s_log_soft, t_soft, reduction='batchmean')
    raw_kl.backward()
    grad_without_t2 = s_logits.grad.clone()

    assert torch.allclose(grad_with_t2, grad_without_t2 * (T ** 2), atol=1e-5)


def test_hinton_kd_zero_temperature_fallback():
    torch.manual_seed(42)
    s_logits = torch.randn(2, 5)
    t_logits = torch.randn(2, 5)
    loss = hinton_kd_loss(s_logits, t_logits, temperature=1.0, alpha=0.5, labels=torch.tensor([1, 0]))
    assert not torch.isnan(loss)
    assert loss.item() >= 0.0
