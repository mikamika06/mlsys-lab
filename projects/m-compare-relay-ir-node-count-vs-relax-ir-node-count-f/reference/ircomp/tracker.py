def count_relay_nodes(model):
    visited = set()

    def visit(node):
        nid = id(node)
        if nid in visited:
            return 0
        visited.add(nid)
        count = 1
        ntype = node.get("type")
        if ntype == "Function":
            count += visit(node["body"])
        elif ntype == "Call":
            count += visit(node["op"])
            for arg in node["args"]:
                count += visit(arg)
        elif ntype == "Tuple":
            for field in node["fields"]:
                count += visit(field)
        elif ntype == "Let":
            count += visit(node["var"])
            count += visit(node["value"])
            count += visit(node["body"])
        return count

    return visit(model["body"])


def count_relax_nodes(model):
    visited = set()

    def visit(node):
        nid = id(node)
        if nid in visited:
            return 0
        visited.add(nid)
        count = 1
        ntype = node.get("type")
        if ntype == "Function":
            for block in node.get("blocks", []):
                count += visit(block)
            if "ret" in node:
                count += visit(node["ret"])
        elif ntype == "DataflowBlock":
            for binding in node.get("bindings", []):
                count += visit(binding)
        elif ntype == "Binding":
            count += visit(node["var"])
            count += visit(node["value"])
        elif ntype == "Call":
            count += visit(node["op"])
            for arg in node["args"]:
                count += visit(arg)
        elif ntype == "Tuple":
            for field in node["fields"]:
                count += visit(field)
        return count

    return visit(model["body"])


def compare_ir_node_counts(model):
    relay_cnt = count_relay_nodes(model["relay"])
    relax_cnt = count_relax_nodes(model["relax"])
    return {
        "relay_nodes": relay_cnt,
        "relax_nodes": relax_cnt,
        "ratio": relax_cnt / relay_cnt if relay_cnt > 0 else 0.0,
    }
