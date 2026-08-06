import re


def build_moe_inventory(tensor_names):
    """Build stacked expert tensor inventory for MoE."""
    inventory = {}
    pattern = re.compile(
        r"model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(.+)"
    )
    for name in tensor_names:
        m = pattern.match(name)
        if m:
            layer_idx, expert_idx, sub_name = m.groups()
            key = (int(layer_idx), sub_name)
            inventory.setdefault(key, []).append(int(expert_idx))
    result = []
    for (layer_idx, sub_name), experts in sorted(inventory.items()):
        result.append(
            {
                "layer": layer_idx,
                "sub": sub_name,
                "experts": sorted(experts),
            }
        )
    return result
