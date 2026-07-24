import numpy as np

FIRST_LEVEL_BLOCK_SIZE = 64  # NF4 weight-quantization block size these absmax values came from


def double_quantize_absmax(absmax, block_size=256):
    """QLoRA-style double quantization of a first-level absmax array.

    See the task description for the exact scheme: mean-subtract, then
    quantize the centered absmax array to int8 in blocks of `block_size`.

    Returns
    -------
    codes : np.ndarray, int8, shape (N,)
    scales : np.ndarray, float64, shape (ceil(N/block_size),)
    mean : float
    recon : np.ndarray, float64, shape (N,)
    bits_saved_per_param : float
    """
    raise NotImplementedError('your code here')
