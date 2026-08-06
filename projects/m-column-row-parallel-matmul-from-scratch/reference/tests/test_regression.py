import torch
import torch.nn as nn
from tp.operators import column_parallel_matmul, row_parallel_matmul


def test_tp_operators_gradient_correctness():
    torch.manual_seed(42)
    x = torch.randn(2, 4, 8, requires_grad=True)
    w_col = torch.randn(12, 8, requires_grad=True)
    w_row = torch.randn(8, 12, requires_grad=True)

    out1 = column_parallel_matmul(x, w_col)
    out1_act = torch.relu(out1)
    out2 = row_parallel_matmul(out1_act, w_row)

    loss = out2.sum()
    loss.backward()

    ref_w_col = w_col.detach().clone().requires_grad_(True)
    ref_w_row = w_row.detach().clone().requires_grad_(True)
    ref_x = x.detach().clone().requires_grad_(True)

    ref_out1 = torch.matmul(ref_x, ref_w_col.t())
    ref_out1_act = torch.relu(ref_out1)
    ref_out2 = torch.matmul(ref_out1_act, ref_w_row.t())
    ref_loss = ref_out2.sum()
    ref_loss.backward()

    assert torch.allclose(out2, ref_out2, atol=1e-5)
    assert torch.allclose(x.grad, ref_x.grad, atol=1e-5)
    assert torch.allclose(w_col.grad, ref_w_col.grad, atol=1e-5)
    assert torch.allclose(w_row.grad, ref_w_row.grad, atol=1e-5)
