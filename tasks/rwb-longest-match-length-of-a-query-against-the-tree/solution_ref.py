def longest_match(tree, query):
    """Walk the tree matching tokens from query; return the longest prefix match length."""
    node = tree
    count = 0
    for token in query:
        if isinstance(node, dict) and token in node:
            node = node[token]
            count += 1
        else:
            break
    return count
