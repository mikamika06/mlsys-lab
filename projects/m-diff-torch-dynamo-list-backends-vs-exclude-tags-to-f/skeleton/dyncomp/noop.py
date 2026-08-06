import torch


def noop_backend(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
    raise NotImplementedError
