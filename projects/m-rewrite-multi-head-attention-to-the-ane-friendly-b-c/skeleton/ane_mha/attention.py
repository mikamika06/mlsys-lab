import numpy as np


class NaiveMHA:

    def __init__(self, embed_dim: int, num_heads: int, qkv_w: np.ndarray,
        out_w: np.ndarray):
        raise NotImplementedError

    def forward(self, x: np.ndarray) ->np.ndarray:
        raise NotImplementedError


class ANEFriendlyMHA:

    def __init__(self, embed_dim: int, num_heads: int, qkv_w: np.ndarray,
        out_w: np.ndarray):
        raise NotImplementedError

    def forward(self, x: np.ndarray) ->np.ndarray:
        raise NotImplementedError
