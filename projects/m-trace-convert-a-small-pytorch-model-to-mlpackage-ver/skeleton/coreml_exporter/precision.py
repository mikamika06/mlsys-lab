"""Module for float32/float16 precision conversions and size measurement."""


def compare_precisions(model, example_inputs, eval_inputs, base_dir):
    """
    Converts model into float32 and float16 mlpackages in base_dir.
    Returns dict with fp32_size, fp16_size, ratio, fp32_err, fp16_err.
    """
    raise NotImplementedError
