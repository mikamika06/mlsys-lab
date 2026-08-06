import numpy as np

def apply_grad_scaler(gradient, scale_factor=1024.0):
    scaled = gradient * scale_factor
    return scaled / scale_factor
