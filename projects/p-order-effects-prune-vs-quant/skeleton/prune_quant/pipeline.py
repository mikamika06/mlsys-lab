def prune_weights(w, sparsity=0.5):
    raise NotImplementedError


def quantize_weights(w, num_bits=4, preserve_zero=False):
    raise NotImplementedError


def measure_both_orders(w, sparsity=0.5, num_bits=4):
    raise NotImplementedError


def analyze_interaction(w, sparsity=0.5, num_bits=4):
    raise NotImplementedError


def joint_recipe(w, sparsity=0.5, num_bits=4):
    raise NotImplementedError


def measure_gains(w, sparsity=0.5, num_bits=4):
    raise NotImplementedError


def justify_order(layers_dict, sparsity=0.5, num_bits=4):
    raise NotImplementedError


def transfer_recipe(layers_dict, sparsity=0.5, num_bits=4):
    raise NotImplementedError
