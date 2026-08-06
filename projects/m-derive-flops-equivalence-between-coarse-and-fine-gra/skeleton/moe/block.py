import numpy as np


class MoEBlock:
    def __init__(self, d_model, d_ffn_fine, num_shared, num_routed, top_k):
        raise NotImplementedError

    def route(self, x):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
