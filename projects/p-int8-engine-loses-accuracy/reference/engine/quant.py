import numpy as np

def collect_layer_outputs(fp16_dict, int8_dict):
    return {"fp16": fp16_dict, "int8": int8_dict}

def compute_layer_mse(fp16_dict, int8_dict):
    res = {}
    for k in fp16_dict:
        res[k] = float(np.mean((fp16_dict[k] - int8_dict[k]) ** 2))
    return res

def identify_sensitive_layers(fp16_dict, int8_dict, top_k=2):
    mses = compute_layer_mse(fp16_dict, int8_dict)
    sorted_layers = sorted(mses.keys(), key=lambda k: mses[k], reverse=True)
    return sorted_layers[:top_k]
