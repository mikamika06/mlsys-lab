import numpy as np

def _score_kv_tile(q_block, k_block, v_block, q_start, k_start, tile_size):
    raise NotImplementedError('your code here')

def causal_flash_attention_forward(Q, K, V, tile_size=2):
    raise NotImplementedError('your code here')
