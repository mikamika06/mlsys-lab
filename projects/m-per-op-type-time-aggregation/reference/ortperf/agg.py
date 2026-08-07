def aggregate_op_types(profile):
    res = {}
    for n in profile["nodes"]:
        op = n["op_type"]
        res[op] = res.get(op, 0.0) + n["dur"]
    return res
