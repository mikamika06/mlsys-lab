import numpy as np


def check_target(orig_size, quant_size, orig_acc, quant_acc, max_drop=0.01):
    size_ratio = quant_size / orig_size
    acc_drop = orig_acc - quant_acc
    return bool(size_ratio <= 0.55 and acc_drop <= max_drop)
