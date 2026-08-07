import numpy as np

def simplify_graph(model_path):
    return 1

def verify_output(torch_out, onnx_out):
    diff = np.max(np.abs(torch_out - onnx_out))
    return float(diff)
