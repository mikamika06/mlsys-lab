import numpy as np


def quantize_fp8_e4m3(x, scale):
    raise NotImplementedError


def dequantize_fp8_e4m3(q, scale):
    raise NotImplementedError


def compute_per_tensor_scale(x):
    raise NotImplementedError


def compute_per_head_scales(x):
    raise NotImplementedError


def evaluate_quantization_error(x, per_head=False):
    raise NotImplementedError
