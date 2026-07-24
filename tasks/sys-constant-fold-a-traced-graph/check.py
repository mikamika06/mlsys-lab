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

def _ref(graph: dict) -> dict:
    """Reference constant-fold implementation (exact same algorithm expected)."""
    g = copy.deepcopy(graph)
    nodes = g["nodes"]
    for nid in sorted(nodes.keys()):          # topological order guaranteed by id increase
        node = nodes[nid]
        op = node["op"]
        if op in ("constant", "input"):
            continue
        # check if all input nodes are already constant (have a 'value' key)
        all_const = True
        for inp_id in node["inputs"]:
            inp_node = nodes[inp_id]
            if "value" not in inp_node:
                all_const = False
                break
        if not all_const:
            continue
        # all constant => compute result
        vals = [nodes[i]["value"] for i in node["inputs"]]
        res = _single_op(op, *vals)
        # replace node with constant
        node.clear()
        node["op"] = "constant"
        node["value"] = res
        node["inputs"] = []
    return g

# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

TEST_GRAPHS: list[dict] = [
    # 0 – trivial all-constant
    {
        "nodes": {
            0: {"op": "constant", "inputs": [], "value": 2},
            1: {"op": "constant", "inputs": [], "value": 3},
            2: {"op": "add", "inputs": [0, 1]},
            3: {"op": "sub", "inputs": [2, 1]},
        },
        "output": 3,
    },
    # 1 – mixed with input (no folding)
    {
        "nodes": {
            0: {"op": "input", "inputs": [], "name": "x"},
            1: {"op": "constant", "inputs": [], "value": 10},
            2: {"op": "mul", "inputs": [0, 1]},
            3: {"op": "constant", "inputs": [], "value": 5},
            4: {"op": "add", "inputs": [2, 3]},
        },
        "output": 4,
    },
    # 2 – chains with partial folding
    {
        "nodes": {
            0: {"op": "constant", "inputs": [], "value": 7},
            1: {"op": "constant", "inputs": [], "value": 2},
            2: {"op": "add", "inputs": [0, 1]},     # 9
            3: {"op": "input", "inputs": [], "name": "y"},
            4: {"op": "mul", "inputs": [2, 3]},     # 9*y  (2 is constant, 3 is input)
            5: {"op": "constant", "inputs": [], "value": 4},
            6: {"op": "add", "inputs": [4, 5]},     # 9*y+4
        },
        "output": 6,
    },
    # 3 – unary negate
    {
        "nodes": {
            0: {"op": "constant", "inputs": [], "value": 5},
            1: {"op": "neg", "inputs": [0]},
        },
        "output": 1,
    },
    # 4 – constant multiplication chain (two levels)
    {
        "nodes": {
            0: {"op": "constant", "inputs": [], "value": 3},
            1: {"op": "constant", "inputs": [], "value": 4},
            2: {"op": "mul", "inputs": [0, 1]},     # 12
            3: {"op": "constant", "inputs": [], "value": 2},
            4: {"op": "mul", "inputs": [2, 3]},     # 24
        },
        "output": 4,
    },
    # 5 – no foldable node at all
    {
        "nodes": {
            0: {"op": "input", "inputs": [], "name": "a"},
            1: {"op": "input", "inputs": [], "name": "b"},
            2: {"op": "add", "inputs": [0, 1]},
        },
        "output": 2,
    },
]

def grade(sol: Any, _fx: Any = None) -> dict:
    for test_id, test_in in enumerate(TEST_GRAPHS):
        # compute reference
        try:
            ref_out = _ref(test_in)
        except Exception:
            return {"exact_match": 0.0}

        # student solution – provide a copy to avoid mutation issues
        try:
            stu_in = copy.deepcopy(test_in)
            stu_out = sol.constant_fold(stu_in)
        except Exception:
            return {"exact_match": 0.0}

        # deep structural comparison
        if not _deep_equal(stu_out, ref_out):
            return {"exact_match": 0.0}

    return {"exact_match": 1.0}

def _deep_equal(a: Any, b: Any) -> bool:
    """Recursive equality that works for our nested dict/list structure."""
    if type(a) != type(b):
        return False
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return False
        for k in a:
            if not _deep_equal(a[k], b[k]):
                return False
        return True
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all(_deep_equal(x, y) for x, y in zip(a, b))
    return a == b
