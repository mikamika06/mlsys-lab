import numpy as np


def merge_online_softmax(m_a, l_a, o_a, m_b, l_b, o_b):
    """
    Merge two online softmax statistics (m_a, l_a, o_a) and (m_b, l_b, o_b).
    Returns (m_merged, l_merged, o_merged).
    """
    raise NotImplementedError


def chunked_online_attention(q, k, v, chunk_size=64):
    """
    Compute attention via chunking and online softmax merging over K/V chunks.
    """
    raise NotImplementedError
