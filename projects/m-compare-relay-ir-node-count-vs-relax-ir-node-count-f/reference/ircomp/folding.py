import copy
import numpy as np


def _eval_op(op, args):
    if op == "add":
        return args[0] + args[1]
    if op == "mul":
        return args[0] * args[1]
    if op == "matmul":
        return np.matmul(args[0], args[1])
    raise ValueError(f"Unknown op: {op}")


def fold_relay_constants(model):
    new_model = copy.deepcopy(model)

    def visit(node):
        ntype = node.get("type")
        if ntype == "Constant":
            return node
        if ntype == "Var":
            return node
        if ntype == "Call":
            node["op"] = visit(node["op"])
            node["args"] = [visit(arg) for arg in node["args"]]
            if all(arg.get("type") == "Constant" for arg in node["args"]):
                op_name = node["op"]["name"]
                vals = [np.array(arg["value"]) for arg in node["args"]]
                res = _eval_op(op_name, vals)
                return {"type": "Constant", "value": res.tolist()}
            return node
        if ntype == "Function":
            node["body"] = visit(node["body"])
            return node
        return node

    new_model["body"] = visit(new_model["body"])
    return new_model


def fold_relax_constants(model):
    new_model = copy.deepcopy(model)
    func = new_model["body"]
    var_map = {}

    for block in func.get("blocks", []):
        new_bindings = []
        for binding in block.get("bindings", []):
            val = binding["value"]
            if val.get("type") == "Call":
                resolved_args = []
                all_const = True
                for arg in val["args"]:
                    if arg.get("type") == "Constant":
                        resolved_args.append(np.array(arg["value"]))
                    elif arg.get("type") == "Var" and arg["name"] in var_map:
                        resolved_args.append(var_map[arg["name"]])
                    else:
                        all_const = False
                        break
                if all_const:
                    op_name = val["op"]["name"]
                    res = _eval_op(op_name, resolved_args)
                    var_map[binding["var"]["name"]] = res
                    binding["value"] = {"type": "Constant", "value": res.tolist()}
            new_bindings.append(binding)
        block["bindings"] = new_bindings
    return new_model


def evaluate_folding_discrepancy(model):
    from ircomp.tracker import count_relax_nodes, count_relay_nodes

    relay_orig = model["relay"]
    relax_orig = model["relax"]

    relay_folded = fold_relay_constants(relay_orig)
    relax_folded = fold_relax_constants(relax_orig)

    r_orig_cnt = count_relay_nodes(relay_orig)
    r_fold_cnt = count_relay_nodes(relay_folded)

    x_orig_cnt = count_relax_nodes(relax_orig)
    x_fold_cnt = count_relax_nodes(relax_folded)

    relay_delta = r_orig_cnt - r_fold_cnt
    relax_delta = x_orig_cnt - x_fold_cnt

    return {
        "relay_delta": relay_delta,
        "relax_delta": relax_delta,
        "discrepancy": relay_delta != relax_delta,
    }
