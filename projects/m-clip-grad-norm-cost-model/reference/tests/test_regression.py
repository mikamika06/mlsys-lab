import torch
from gradclip.clip import compute_grad_norm, clip_grads


def test_grad_norm_basic():
    p = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    p.grad = torch.tensor([3.0, 4.0])
    norm = compute_grad_norm([p], norm_type=2.0)
    assert abs(norm.item() - 5.0) < 1e-5


def test_clip_scaling():
    p = torch.nn.Parameter(torch.tensor([10.0, 10.0]))
    p.grad = torch.tensor([6.0, 8.0])
    clip_grads([p], max_norm=5.0, norm_type=2.0)
    assert abs(p.grad[0].item() - 3.0) < 1e-5
    assert abs(p.grad[1].item() - 4.0) < 1e-5
