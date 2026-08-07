import numpy as np

def get_sample_log():
    return "Error: Invalid Node\nError: Shape Mismatch\nError: Undefined Symbol"

def get_sample_tensors():
    x = np.ones((4, 16), dtype=np.float32)
    y = np.ones((4, 16), dtype=np.float32)
    return x, y
