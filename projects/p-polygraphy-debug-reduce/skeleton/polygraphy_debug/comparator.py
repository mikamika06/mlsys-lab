import numpy as np


class StepComparator:
    def __init__(self, ref_runner, target_runner, rtol=1e-3, atol=1e-4):
        raise NotImplementedError

    def compare_layer(self, layer_name, inputs):
        raise NotImplementedError

    def find_first_divergence(self, execution_graph, inputs):
        raise NotImplementedError
