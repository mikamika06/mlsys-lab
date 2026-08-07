import torch
from torch.utils.flop_counter import FlopCounterMode

def measure_flops(model, x):
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        model(x)
    return flop_counter.get_total_flops()
