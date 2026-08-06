import torch


def noop_backend(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
    """A custom no-op backend that returns the graph module forward method."""
    return gm.forward
