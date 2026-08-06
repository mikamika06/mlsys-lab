import numpy as np


def optimal_wrap_threshold(total_params, num_layers, communication_cost_factor):
    avg_layer_size = total_params / num_layers
    optimal_threshold = np.sqrt(communication_cost_factor * avg_layer_size)
    return float(optimal_threshold)
