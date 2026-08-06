import numpy as np


def quantize_fp16_to_int8(weight_fp16):
    raise NotImplementedError


def pack_prequantized_artifact(weight_fp16):
    raise NotImplementedError


def load_model_weights(raw_weights, mode="inflight"):
    raise NotImplementedError
