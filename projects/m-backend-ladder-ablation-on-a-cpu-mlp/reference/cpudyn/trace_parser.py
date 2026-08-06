import re


def rebuild_op_sequence(graph_code_log):
    """Rebuild ordered list of torch operations from a graph_code log string."""
    ops = []
    lines = graph_code_log.splitlines()
    for line in lines:
        line = line.strip()
        match = re.search(r'=\s*(torch\.[a-zA-Z0-9_.]+)\(', line)
        if match:
            ops.append(match.group(1))
        else:
            match_call = re.search(r'=\s*([a-zA-Z0-9_.]+)\(', line)
            if match_call and match_call.group(1).startswith("torch."):
                ops.append(match_call.group(1))
    return ops
