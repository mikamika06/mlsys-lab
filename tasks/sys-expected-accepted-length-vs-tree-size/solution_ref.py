def expected_accepted_length(tree, accept_prob):
    def visit(node, path_prob):
        total = 0.0
        for child in tree[node]:
            child_prob = path_prob * float(accept_prob[child])
            total += child_prob
            total += visit(child, child_prob)
        return total

    return float(visit(0, 1.0))
