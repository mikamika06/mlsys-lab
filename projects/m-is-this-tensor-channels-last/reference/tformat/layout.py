import torch


def is_channels_last(tensor: torch.Tensor) -> bool:
    if tensor.dim() != 4:
        return False
    return tensor.is_contiguous(memory_format=torch.channels_last)


def compute_nhwc_strides(shape: tuple) -> tuple:
    n, c, h, w = shape
    return (h * w * c, 1, w * c, c)


def analyze_transfer_block(is_pinned: bool, is_contiguous: bool) -> str:
    if not is_pinned:
        return "fallback_to_sync"
    if not is_contiguous:
        return "non_contiguous_blocking"
    return "optimal_overlap"
