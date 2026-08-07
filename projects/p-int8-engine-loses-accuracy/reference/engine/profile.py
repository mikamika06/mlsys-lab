import numpy as np

def profile_layers(model, fp16_outputs, int8_outputs):
    res = {}
    for i, (f_out, i_out) in enumerate(zip(fp16_outputs, int8_outputs)):
        mse = float(np.mean((f_out - i_out) ** 2))
        res[f"layer_{i}"] = mse
    return res

def identify_sensitive_layers(layer_mses, top_k=3):
    indexed = list(enumerate(layer_mses))
    indexed.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in indexed[:top_k]]
