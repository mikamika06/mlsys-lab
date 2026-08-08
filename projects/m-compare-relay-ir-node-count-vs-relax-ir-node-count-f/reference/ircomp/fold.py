import copy
from ircomp.nodes import count_relay_nodes, count_relax_nodes


def fold_relay_constants(ast_dict):
    """Simulate constant folding pass on Relay AST and return folded AST and node count."""
    tree = copy.deepcopy(ast_dict)

    def _fold(node):
        if not isinstance(node, dict):
            return node
        ntype = node.get("type")
        if ntype == "Function":
            node["body"] = _fold(node["body"])
            return node
        if ntype == "Call":
            node["op"] = _fold(node["op"])
            node["args"] = [_fold(a) for a in node.get("args", [])]
            args = node["args"]
            op_name = node.get("op", {}).get("name")
            if all(a.get("type") == "Constant" for a in args):
                vals = [a.get("value") for a in args]
                if op_name == "add":
                    res = vals[0] + vals[1]
                elif op_name == "multiply":
                    res = vals[0] * vals[1]
                elif op_name == "relu":
                    res = max(0, vals[0])
                elif op_name == "bias_add":
                    res = vals[0] + vals[1]
                else:
                    res = vals[0]
                return {"type": "Constant", "value": res}
            return node
        return node

    folded = _fold(tree)
    return folded, count_relay_nodes(folded)


def fold_relax_constants(ast_dict):
    """Simulate constant folding pass on Relax AST and return folded AST and node count."""
    tree = copy.deepcopy(ast_dict)
    const_env = {}

    def _fold_expr(expr):
        if not isinstance(expr, dict):
            return expr
        ntype = expr.get("type")
        if ntype == "Var":
            vid = expr.get("name")
            if vid in const_env:
                return const_env[vid]
            return expr
        if ntype == "Call":
            expr["op"] = _fold_expr(expr["op"])
            expr["args"] = [_fold_expr(a) for a in expr.get("args", [])]
            args = expr["args"]
            op_name = expr.get("op", {}).get("name")
            if all(a.get("type") == "Constant" for a in args):
                vals = [a.get("value") for a in args]
                if op_name == "add":
                    res = vals[0] + vals[1]
                elif op_name == "multiply":
                    res = vals[0] * vals[1]
                elif op_name == "relu":
                    res = max(0, vals[0])
                elif op_name == "bias_add":
                    res = vals[0] + vals[1]
                else:
                    res = vals[0]
                return {"type": "Constant", "value": res, "sinfo": expr.get("sinfo")}
            return expr
        return expr

    def _visit(node):
        if not isinstance(node, dict):
            return node
        ntype = node.get("type")
        if ntype == "Function":
            for b in node.get("blocks", []):
                _visit(b)
            node["body"] = _fold_expr(node.get("body", {}))
            return node
        if ntype in ("BindingBlock", "DataflowBlock"):
            new_bindings = []
            for b in node.get("bindings", []):
                b["value"] = _fold_expr(b.get("value", {}))
                var_name = b.get("var", {}).get("name")
                if b["value"].get("type") == "Constant":
                    const_env[var_name] = b["value"]
                new_bindings.append(b)
            node["bindings"] = new_bindings
            return node
        return node

    folded = _visit(tree)
    return folded, count_relax_nodes(folded)


def analyze_folding_divergence(subgraphs):
    """Compute node reduction counts and divergence ratios between Relay and Relax."""
    results = []
    for sg in subgraphs:
        r_orig = count_relay_nodes(sg["relay"])
        _, r_fold = fold_relay_constants(sg["relay"])
        r_red = r_orig - r_fold

        x_orig = count_relax_nodes(sg["relax"])
        _, x_fold = fold_relax_constants(sg["relax"])
        x_red = x_orig - x_fold

        div_ratio = (r_red / x_red) if x_red != 0 else 0.0
        results.append({
            "name": sg["name"],
            "relay_reduction": r_red,
            "relax_reduction": x_red,
            "divergence_ratio": round(div_ratio, 4)
        })
    return results
