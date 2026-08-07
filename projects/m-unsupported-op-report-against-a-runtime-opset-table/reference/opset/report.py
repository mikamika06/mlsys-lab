def check_support(nodes, model_opset, ops_table):
    unsupported = []
    for n in nodes:
        op = n["op_type"]
        if op not in ops_table:
            unsupported.append(n["name"])
        elif model_opset > ops_table[op]:
            unsupported.append(n["name"])
    return unsupported
