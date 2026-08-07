def min_gpu_memory(layer_bytes: list[int], K: int, activation_buffer: int) -> int:
    """
    Minimum GPU memory (bytes) to run a layer-streamed model: the heaviest
    K-layer sliding-window sum of `layer_bytes`, plus `activation_buffer`.
    See task.md for the exact formula.
    """
    raise NotImplementedError('your code here')
