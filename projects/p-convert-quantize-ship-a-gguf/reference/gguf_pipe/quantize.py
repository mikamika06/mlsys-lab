import json
import os


def get_quantized_size(model_path, recipe):
    base_size = 1000000000
    multipliers = {"Q4_K_M": 0.35, "Q8_0": 0.55, "FP16": 1.0}
    return int(base_size * multipliers.get(recipe, 0.5))


def generate_imatrix(model_path, calibration_data_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = {"imatrix": True, "source": model_path}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return output_path


def quantize_model(input_path, output_path, recipe, imatrix_path=None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = {"recipe": recipe, "imatrix": imatrix_path, "input": input_path}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return output_path
