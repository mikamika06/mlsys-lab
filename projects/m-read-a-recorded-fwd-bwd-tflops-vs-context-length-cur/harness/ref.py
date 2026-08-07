import numpy as np

def get_test_data():
    lengths = [256, 512, 1024, 2048, 4096, 8192]
    base = [120.0, 140.0, 180.0, 220.0, 250.0, 260.0]
    cust = [90.0, 130.0, 190.0, 240.0, 270.0, 280.0]
    return lengths, cust, base

def find_crossover_length(lengths, custom_tflops, baseline_tflops):
    for l, c, b in zip(lengths, custom_tflops, baseline_tflops):
        if c >= b:
            return int(l)
    return int(lengths[-1])

def rescale_block(running_max, running_sum, block_max, block_sum, use_old_max=False):
    if use_old_max:
        scaled_running = running_sum
        scaled_block = block_sum * np.exp(block_max - running_max)
        return running_max, scaled_running + scaled_block
    new_max = max(running_max, block_max)
    scaled_running = running_sum * np.exp(running_max - new_max)
    scaled_block = block_sum * np.exp(block_max - new_max)
    return new_max, scaled_running + scaled_block
