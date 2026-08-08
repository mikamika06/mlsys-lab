import numpy as np

def get_mock_models():
    def forward_fp16(layer, dataset):
        return np.array(dataset, dtype=np.float32)

    def forward_int8(layer, dataset):
        arr = np.array(dataset, dtype=np.float32)
        if layer == "sensitive_layer":
            return arr * 0.85
        return arr * 0.99

    model_fp16 = {"layers": ["safe_layer", "sensitive_layer"], "forward": forward_fp16}
    model_int8 = {"layers": ["safe_layer", "sensitive_layer"], "forward": forward_int8}
    return model_fp16, model_int8
