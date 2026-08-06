import torch


def is_channels_last(tensor: torch.Tensor) -> bool:
    raise NotImplementedError


def compute_nhwc_strides(shape: tuple) -> tuple:
    raise NotImplementedError


def analyze_transfer_block(is_pinned: bool, is_contiguous: bool) -> str:
    raise NotImplementedError
