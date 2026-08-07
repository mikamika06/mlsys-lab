import torch


def fine_tune_recovery(model: torch.nn.Module, dataloader, epochs: int, lr: float) -> torch.nn.Module:
    raise NotImplementedError
