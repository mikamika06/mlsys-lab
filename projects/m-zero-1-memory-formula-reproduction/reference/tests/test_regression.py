import torch
from zero1.memory import calculate_zero1_memory
from zero1.distributed import ToyZeRO1Optimizer


def test_zero1_memory_sharding_invariant():
    num_params = 1000000
    world_size = 4
    mem = calculate_zero1_memory(num_params, world_size, precision_bytes=2)

    total_opt_unsharded = num_params * 12
    expected_opt_per_rank = total_opt_unsharded / world_size

    assert mem["opt_state_per_rank_bytes"] == expected_opt_per_rank
    assert mem["zero1_bytes"] < mem["baseline_bytes"]


def test_toy_zero1_step_updates_params():
    params = [torch.nn.Parameter(torch.ones(10, 10, dtype=torch.float16))]
    world_size = 2

    opt0 = ToyZeRO1Optimizer(params, lr=1e-2, world_size=world_size, rank=0)
    opt1 = ToyZeRO1Optimizer(params, lr=1e-2, world_size=world_size, rank=1)

    assert opt0.get_rank_state_bytes() == 50 * 3 * 4
    assert opt1.get_rank_state_bytes() == 50 * 3 * 4

    grads = [torch.ones(10, 10, dtype=torch.float16)]
    opt0.step(grads)
    opt1.step(grads)

    assert not torch.allclose(params[0], torch.ones(10, 10, dtype=torch.float16))
