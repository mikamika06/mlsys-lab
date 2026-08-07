"""Reconstruct canonical bottom-up fully_shard sequence."""


def reconstruct_fully_shard_sequence(model_tree):
    """Return ordered list of module paths for valid bottom-up fully_shard calls."""
    sequence = []

    def post_order(node, path):
        for child_name, child_node in node.get("children", {}).items():
            child_path = f"{path}.{child_name}" if path else child_name
            post_order(child_node, child_path)

        if node.get("should_shard", True):
            sequence.append(path)

    post_order(model_tree, "")
    return sequence
