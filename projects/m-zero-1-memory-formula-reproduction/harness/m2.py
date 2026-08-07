import ref
import torch


def check(workdir):
    from zero1.toy_zero import partition_optimizer_states

    p1 = torch.nn.Parameter(torch.randn(100))
    p2 = torch.nn.Parameter(torch.randn(100))
    params = [p1, p2]

    got = partition_optimizer_states(params, 2, 0)
    want = ref.oracle_partition(params, 2, 0)

    match = 1.0 if len(got) == len(want) else 0.0
    return {"shard_states_matched": match}
