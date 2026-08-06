"""Sharding and load balancing utilities for FSDP1."""


def compute_load_balance(param_sizes, world_size, strategy):
    if strategy == "per_tensor":
        ranks = [0] * world_size
        sorted_indices = sorted(range(len(param_sizes)), key=lambda i: param_sizes[i], reverse=True)
        for idx in sorted_indices:
            min_r = min(range(world_size), key=lambda r: ranks[r])
            ranks[min_r] += param_sizes[idx]
        max_load = max(ranks)
        min_load = min(ranks)
        return max_load / (min_load + 1e-9)
    elif strategy == "flat":
        total = sum(param_sizes)
        chunk = (total + world_size - 1) // world_size
        ranks = [chunk] * world_size
        max_load = max(ranks)
        min_load = min(ranks)
        return max_load / (min_load + 1e-9)
    else:
        raise ValueError("unknown strategy")


def auto_wrap_assign(module_tree, min_params):
    units = []

    def recurse(node):
        total = node["params"]
        for child in node["children"]:
            total += recurse(child)
        if total >= min_params:
            units.append(node["name"])
            return 0
        return total

    recurse(module_tree)
    if not units and module_tree["name"]:
        units.append(module_tree["name"])
    return sorted(units)


def check_freeze_constraint(flat_param_size, frozen_size):
    if frozen_size > 0 and frozen_size < flat_param_size:
        raise RuntimeError("Freeze constraint violated: partial freezing within a FlatParameter is not supported.")
    return True
