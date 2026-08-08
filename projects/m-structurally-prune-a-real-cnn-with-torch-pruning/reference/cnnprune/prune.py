import numpy as np


class CNNModel:
    def __init__(self):
        self.conv1_out = 16
        self.conv2_in = 16
        self.conv2_out = 16
        self.fc_in = 16 * 32 * 32


def structural_prune(model, prune_ratio):
    keep = int(model.conv1_out * (1.0 - prune_ratio))
    model.conv1_out = keep
    model.conv2_in = keep
    model.conv2_out = keep
    model.fc_in = keep * 32 * 32
    return model


def measure_speedup(model_orig, model_pruned, sample_input):
    orig_params = model_orig.conv1_out * 3 * 3 * 3 + model_orig.conv2_out * model_orig.conv2_in * 3 * 3 + model_orig.fc_in * 10
    pruned_params = model_pruned.conv1_out * 3 * 3 * 3 + model_pruned.conv2_out * model_pruned.conv2_in * 3 * 3 + model_pruned.fc_in * 10
    return float(pruned_params) / float(orig_params)
