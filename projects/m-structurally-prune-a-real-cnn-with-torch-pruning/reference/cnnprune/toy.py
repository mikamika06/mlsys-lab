import numpy as np


class ToyNetSimulator:
    def __init__(self):
        self.conv1_out_channels = 8
        self.conv2_in_channels = 8


def propagate_channels(pruned_channels):
    groups = {
        "conv1_out": pruned_channels,
        "bn1": pruned_channels,
        "conv2_in": pruned_channels
    }
    return groups
