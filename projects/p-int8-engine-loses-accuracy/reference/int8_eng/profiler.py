import numpy as np

def profile_layers(model_fp16, model_int8, dataset):
    res = {}
    for name in model_fp16["layers"]:
        out_fp = model_fp16["forward"](name, dataset)
        out_int = model_int8["forward"](name, dataset)
        diff = np.mean(np.abs(out_fp - out_int))
        res[name] = float(diff)
    return res
