def ov_quantizer_one_shot(model_weights, calibration_data, config):
    """Perform one-shot OVQuantizer execution."""
    raise NotImplementedError


def manual_nncf_quantize(model_weights, calibration_data, config):
    """Perform manual NNCF quantization pipeline."""
    raise NotImplementedError


def verify_output_parity(weights, inputs, config):
    """Compare outputs of one-shot vs manual quantization pipelines."""
    raise NotImplementedError
