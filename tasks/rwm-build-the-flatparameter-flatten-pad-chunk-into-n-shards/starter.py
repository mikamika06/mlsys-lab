import numpy as np


def flatten_pad_shard(params: list[np.ndarray], world_size: int) -> list[np.ndarray]:
    """FSDP-style FlatParameter: flatten, zero-pad, chunk into N shards.

    params: list of arrays of arbitrary shape (a module's parameters, in
        order).
    world_size: number of shards (ranks), N.

    Steps:
      1. Ravel every array in `params` (in order) and concatenate into one
         1-D float64 buffer of length `total`.
      2. Zero-pad the buffer at the end to the next multiple of
         `world_size` (0 extra elements if `total` is already a multiple).
      3. Split the padded buffer into `world_size` equal-length
         contiguous shards, each of length `padded_total / world_size`.

    Returns a list of `world_size` 1-D float64 arrays, all the same
    length.
    """
    raise NotImplementedError('your code here')
