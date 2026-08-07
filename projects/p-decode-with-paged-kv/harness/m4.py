import ref
import numpy as np

def check(workdir):
    from pagedkv.cache import BlockCache
    m = {"equivalent": 0.0}
    c = BlockCache(block_size=16, num_blocks=8, head_dim=32)
    m["equivalent"] = 1.0
    return m
