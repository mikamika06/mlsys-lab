import numpy as np

MODEL_SPECS = [
    {"param_sizes": [1000, 2000, 1500, 500, 3000, 1000], "world_size": 2},
    {"param_sizes": [500, 500, 500, 500, 500, 500, 500, 500], "world_size": 4},
    {"param_sizes": [1024, 2048, 512, 4096, 128], "world_size": 2},
]

MODULE_TREES = [
    {"name": "root", "params": 100, "children": [
        {"name": "layer1", "params": 500, "children": []},
        {"name": "layer2", "params": 600, "children": []}
    ]},
    {"name": "root", "params": 50, "children": [
        {"name": "block1", "params": 200, "children": [
            {"name": "sub1", "params": 150, "children": []}
        ]},
        {"name": "block2", "params": 300, "children": []}
    ]},
    {"name": "root", "params": 10, "children": [
        {"name": "a", "params": 1000, "children": []},
        {"name": "b", "params": 1000, "children": []}
    ]}
]


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
