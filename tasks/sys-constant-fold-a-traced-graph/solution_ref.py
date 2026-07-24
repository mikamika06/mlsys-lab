from __future__ import annotations

import copy
from typing import Any

def _single_op(op: str, *args: int) -> int:
    if op == "add":
        return args[0] + args[1]
    if op == "mul":
        return args[0] * args[1]
    if op == "sub":
        return args[0] - args[1]
    if op == "neg":
        return -args[0]
    raise ValueError(f"unknown op {op}")

def constant_fold(graph: dict) -> dict:
    """Perform constant folding on a traced integer graph.

    See task.md for the exact specification.
    """
    g = copy.deepcopy(graph)
    nodes = g["nodes"]

    for nid in sorted(nodes.keys()):          # topological order (id increasing)
        node = nodes[nid]
        op = node["op"]

        if op in ("constant", "input"):
            continue

        # check if all inputs are already constant (have a 'value' key)
        all_const = True
        for inp_id in node["inputs"]:
            inp_node = nodes[inp_id]
            if "value" not in inp_node:
                all_const = False
                break

        if not all_const:
            continue

        # compute result
        vals = [nodes[i]["value"] for i in node["inputs"]]
        res = _single_op(op, *vals)

        # replace node with constant
        node.clear()
        node["op"] = "constant"
        node["value"] = res
        node["inputs"] = []

    return g
