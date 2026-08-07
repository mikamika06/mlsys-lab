import numpy as np

def rescale_block(running_max, running_sum, block_max, block_sum, use_old_max=False):
    if use_old_max:
        scaled_running = running_sum
        scaled_block = block_sum * np.exp(block_max - running_max)
        return running_max, scaled_running + scaled_block
    new_max = max(running_max, block_max)
    scaled_running = running_sum * np.exp(running_max - new_max)
    scaled_block = block_sum * np.exp(block_max - new_max)
    return new_max, scaled_running + scaled_block
