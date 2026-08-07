def get_op_diff(ops_a, ops_b):
    return sorted(list(set(ops_a) ^ set(ops_b)))
