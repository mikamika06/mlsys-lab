import numpy as np

FP16_MAX = 65504.0


def locate_fp16_overflow_layers(graph):
    nan_layers = []
    for node in graph["nodes"]:
        for out_name in node["outputs"]:
            act = graph["activations"].get(out_name)
            if act is not None:
                if np.any(np.isnan(act)) or np.any(np.isinf(act)) or np.any(np.abs(act) > FP16_MAX):
                    nan_layers.append(node["name"])
                    break
    return nan_layers
