import numpy as np


def rewrite_unfold(x, kernel_size, stride, padding, dilation):
    batch, channels, length = x.shape
    out_length = (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    padded = np.pad(x, ((0, 0), (0, 0), (padding, padding)), mode="constant")
    patches = []
    for i in range(out_length):
        start = i * stride
        chunk = padded[:, :, start:start + dilation * kernel_size:dilation]
        patches.append(chunk)
    stacked = np.stack(patches, axis=-1)
    reshaped = stacked.reshape(batch, channels * kernel_size, out_length)
    return reshaped
