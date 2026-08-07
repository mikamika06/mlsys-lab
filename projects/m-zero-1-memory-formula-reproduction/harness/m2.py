import torch


def check(workdir):
    from zero1.distributed import ToyZeRO1Optimizer

    out = {"sharding_correct": 0.0, "step_correct": 0.0}

    params_r0 = [torch.nn.Parameter(torch.ones(10, 10, dtype=torch.float16))]
    params_r1 = [torch.nn.Parameter(torch.ones(10, 10, dtype=torch.float16))]

    opt0 = ToyZeRO1Optimizer(params_r0, lr=0.1, world_size=2, rank=0)
    opt1 = ToyZeRO1Optimizer(params_r1, lr=0.1, world_size=2, rank=1)

    b0 = opt0.get_rank_state_bytes()
    b1 = opt1.get_rank_state_bytes()

    expected_bytes_per_rank = 50 * 3 * 4
    if b0 == expected_bytes_per_rank and b1 == expected_bytes_per_rank:
        out["sharding_correct"] = 1.0

    grads_r0 = [torch.ones(10, 10, dtype=torch.float16)]
    grads_r1 = [torch.ones(10, 10, dtype=torch.float16)]

    opt0.step(grads_r0)
    opt1.step(grads_r1)

    p0_updated = params_r0[0].detach()
    p1_updated = params_r1[0].detach()

    ones = torch.ones(10, 10, dtype=torch.float16)
    if (
        not torch.allclose(p0_updated, ones)
        and torch.allclose(p0_updated, p1_updated)
        and p0_updated.dtype == torch.float16
    ):
        out["step_correct"] = 1.0

    return out
