import numpy as np


class MemoryTracker:

    def __init__(self):
        self.reads = 0
        self.writes = 0

    def reset(self):
        self.reads = 0
        self.writes = 0

    def track_read(self, size_elements):
        raise NotImplementedError

    def track_write(self, size_elements):
        raise NotImplementedError


def online_softmax_update(m_prev, l_prev, acc_prev, S_block, V_block):
    raise NotImplementedError


def naive_attention(Q, K, V, sm_scale=1.0, is_causal=False):
    raise NotImplementedError
