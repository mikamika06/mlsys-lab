import numpy as np

def get_reference_data():
    doc_ids = [0, 0, 0, 0, 1, 1, 1, 1]
    window_size = 2
    block_size = 2
    seq_len = 8
    return doc_ids, window_size, block_size, seq_len
