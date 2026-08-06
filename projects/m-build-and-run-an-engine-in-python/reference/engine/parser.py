class ParseError(Exception):
    pass

def parse_model(model_def, supported_ops):
    errors = []
    nodes = model_def.get("nodes", [])
    for idx, node in enumerate(nodes):
        op = node.get("op")
        if op not in supported_ops:
            errors.append(f"Node {idx} op {op} is unsupported")
    if errors:
        raise ParseError("; ".join(errors))
    return {"status": "parsed", "node_count": len(nodes)}
