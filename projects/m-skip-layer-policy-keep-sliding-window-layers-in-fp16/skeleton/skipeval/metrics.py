def quantize_dequantize_fp8(x):
    raise NotImplementedError


def compute_ppl_delta(model_data, dtypes):
    raise NotImplementedError


def passes_accuracy_gate(ppl_delta, max_allowed_delta=0.01):
    raise NotImplementedError
