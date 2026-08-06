import torch
import harness.ref as ref


def check(workdir):
    from tp.operators import column_parallel_matmul, row_parallel_matmul
    from tp.mlp import TensorParallelMLP

    out = {"max_abs_err": 1.0}
    torch.manual_seed(1337)

    batch, seq, hidden_dim, ffn_dim = 2, 8, 16, 32
    x = torch.randn(batch, seq, hidden_dim, requires_grad=True)
    w1 = torch.randn(ffn_dim, hidden_dim, requires_grad=True)
    w2 = torch.randn(hidden_dim, ffn_dim, requires_grad=True)

    mlp = TensorParallelMLP(hidden_dim=hidden_dim, ffn_dim=ffn_dim, process_group=None)
    mlp.w1.data.copy_(w1.data)
    mlp.w2.data.copy_(w2.data)

    y = mlp(x)
    ref_y = ref.compute_ref_mlp(x, w1, w2)

    err1 = (y - ref_y).abs().max().item()

    loss = y.sum()
    loss.backward()

    ref_x = x.detach().clone().requires_grad_(True)
    ref_w1 = w1.detach().clone().requires_grad_(True)
    ref_w2 = w2.detach().clone().requires_grad_(True)
    ref_y_grad = ref.compute_ref_mlp(ref_x, ref_w1, ref_w2)
    ref_y_grad.sum().backward()

    err_gx = (x.grad - ref_x.grad).abs().max().item()
    err_gw1 = (mlp.w1.grad - ref_w1.grad).abs().max().item()
    err_gw2 = (mlp.w2.grad - ref_w2.grad).abs().max().item()

    total_err = max(err1, err_gx, err_gw1, err_gw2)
    out["max_abs_err"] = float(total_err)
    return out
