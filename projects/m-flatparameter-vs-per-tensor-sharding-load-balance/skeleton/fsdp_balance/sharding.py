def compute_load_balance(param_sizes, world_size, strategy):
    raise NotImplementedError


def auto_wrap_assign(module_tree, min_params):
    raise NotImplementedError


def check_freeze_constraint(flat_param_size, frozen_size):
    raise NotImplementedError
