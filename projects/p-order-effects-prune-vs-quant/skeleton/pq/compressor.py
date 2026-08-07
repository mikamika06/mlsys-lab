def prune_weights(w, sparsity):
    raise NotImplementedError


def quantize_weights(w, bits):
    raise NotImplementedError


def evaluate_pipeline(w, x, sparsity, bits, order):
    raise NotImplementedError


def find_joint_recipe(w, x, target_sparsity, target_bits):
    raise NotImplementedError


def measure_gains(w, sparsity, bits, order):
    raise NotImplementedError


def justify_order(w, x, sparsity, bits):
    raise NotImplementedError
