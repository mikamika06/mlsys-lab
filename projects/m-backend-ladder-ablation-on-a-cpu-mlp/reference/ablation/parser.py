def parse_graph_code(code_str):
    ops = []
    for line in code_str.split('\n'):
        line = line.strip()
        if '=' in line and '(' in line:
            rhs = line.split('=', 1)[1].strip()
            op = rhs.split('(')[0].strip()
            if op.startswith("torch.ops."):
                ops.append(op)
    return ops
