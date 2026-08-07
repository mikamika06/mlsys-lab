import numpy as np

def quantize_q8_0(tensor):
    raise NotImplementedError

def dequantize_q8_0(qdata, scales):
    raise NotImplementedError

def quantize_q4_0(tensor):
    raise NotImplementedError

def dequantize_q4_0(qdata, scales):
    raise NotImplementedError

def compute_imatrix(activations):
    raise NotImplementedError

def quantize_imatrix(tensor, imatrix, n_bits=4):
    raise NotImplementedError
