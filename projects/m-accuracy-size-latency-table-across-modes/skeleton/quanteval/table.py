def calculate_layer_size(weights, bias, mode):
    raise NotImplementedError


def calculate_model_size(layers, mode):
    raise NotImplementedError


def estimate_model_latency(layers, mode, hw_config):
    raise NotImplementedError


def build_ptq_summary_table(layers, hw_config, test_inputs):
    raise NotImplementedError


def rank_ptq_modes(table, max_mse_threshold=0.01):
    raise NotImplementedError
