def longest_valid_prefix(tree, target):
    if not target:
        return []

    if tree.get("token") != target[0]:
        return []

    result = [tree["token"]]
    node = tree

    for token in target[1:]:
        next_node = None
        for child in node.get("children", []):
            if child.get("token") == token:
                next_node = child
                break

        if next_node is None:
            break

        result.append(next_node["token"])
        node = next_node

    return result
