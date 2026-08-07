"""Measure wrap order violations in FSDP2 model wrapping."""


def _get_depths_and_children(tree, current_path="", depth=0, depth_map=None, children_map=None):
    if depth_map is None:
        depth_map = {}
    if children_map is None:
        children_map = {}

    depth_map[current_path] = depth
    children = []

    for child_name, child_node in tree.get("children", {}).items():
        child_path = f"{current_path}.{child_name}" if current_path else child_name
        children.append(child_path)
        _get_depths_and_children(child_node, child_path, depth + 1, depth_map, children_map)

    children_map[current_path] = children
    return depth_map, children_map


def analyze_wrap_violations(model_tree, wrap_sequence):
    """Measure bottom-up wrap order violations and penalty units."""
    depth_map, children_map = _get_depths_and_children(model_tree)

    wrapped = set()
    violations = 0
    penalty = 0

    for path in wrap_sequence:
        if path not in depth_map:
            continue

        unwrapped_children = [c for c in children_map.get(path, []) if c not in wrapped]
        if unwrapped_children:
            violations += 1
            for child_path in unwrapped_children:
                child_depth = depth_map.get(child_path, 0)
                penalty += (child_depth + 1) * 10

        wrapped.add(path)

    return {
        "violations_count": violations,
        "penalty_units": penalty,
        "is_valid_bottom_up": violations == 0,
    }
