def count_relay_nodes(node):
    """Count total AST nodes in a Relay IR AST dictionary representation."""
    if not isinstance(node, dict):
        return 0
    ntype = node.get("type")
    if not ntype:
        return 0
    total = 1
    if ntype == "Function":
        for p in node.get("params", []):
            total += count_relay_nodes(p)
        total += count_relay_nodes(node.get("body", {}))
    elif ntype == "Call":
        total += count_relay_nodes(node.get("op", {}))
        for arg in node.get("args", []):
            total += count_relay_nodes(arg)
    elif ntype == "Tuple":
        for field in node.get("fields", []):
            total += count_relay_nodes(field)
    elif ntype == "Let":
        total += count_relay_nodes(node.get("var", {}))
        total += count_relay_nodes(node.get("value", {}))
        total += count_relay_nodes(node.get("body", {}))
    elif ntype in ("Var", "Constant", "Op"):
        pass
    return total


def count_relax_nodes(node):
    """Count total AST nodes in a Relax IR AST dictionary representation."""
    if not isinstance(node, dict):
        return 0
    ntype = node.get("type")
    if not ntype:
        return 0
    total = 1
    if "sinfo" in node and node["sinfo"] is not None:
        total += count_relax_nodes(node["sinfo"])
    if ntype == "Function":
        for p in node.get("params", []):
            total += count_relax_nodes(p)
        for block in node.get("blocks", []):
            total += count_relax_nodes(block)
        total += count_relax_nodes(node.get("body", {}))
    elif ntype in ("BindingBlock", "DataflowBlock"):
        for b in node.get("bindings", []):
            total += count_relax_nodes(b)
    elif ntype == "VarBinding":
        total += count_relax_nodes(node.get("var", {}))
        total += count_relax_nodes(node.get("value", {}))
    elif ntype == "Call":
        total += count_relax_nodes(node.get("op", {}))
        for arg in node.get("args", []):
            total += count_relax_nodes(arg)
    elif ntype == "SeqExpr":
        for block in node.get("blocks", []):
            total += count_relax_nodes(block)
        total += count_relax_nodes(node.get("body", {}))
    elif ntype in ("Var", "Constant", "Op", "TensorStructInfo", "TupleStructInfo"):
        pass
    return total


def compare_node_counts(subgraphs):
    """Return a list of dicts comparing Relay vs Relax node counts for each subgraph."""
    res = []
    for sg in subgraphs:
        relay_c = count_relay_nodes(sg["relay"])
        relax_c = count_relax_nodes(sg["relax"])
        res.append({
            "name": sg["name"],
            "relay_count": relay_c,
            "relax_count": relax_c,
            "diff": relax_c - relay_c
        })
    return res
