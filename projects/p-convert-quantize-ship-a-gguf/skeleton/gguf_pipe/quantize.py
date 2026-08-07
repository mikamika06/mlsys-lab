def get_quantized_size(model_path, recipe):
    raise NotImplementedError


def generate_imatrix(model_path, calibration_data_path, output_path):
    raise NotImplementedError


def quantize_model(input_path, output_path, recipe, imatrix_path=None):
    raise NotImplementedError
