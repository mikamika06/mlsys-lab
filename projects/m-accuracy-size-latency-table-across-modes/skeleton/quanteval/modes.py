def quantize_fp16(arr):
    raise NotImplementedError


def quantize_int8_weights(weights):
    raise NotImplementedError


def quantize_int8_activations(x, rmin, rmax):
    raise NotImplementedError


def evaluate_mode_output(weights, bias, x, mode, calibration_range=None):
    raise NotImplementedError
