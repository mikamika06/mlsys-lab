from typing import List, Dict, Tuple, Any

def _reference(nodes: List[Dict[str, Any]], outputs: List[int]) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Compute the expected result using a simple DFS over the graph."""
    node_map = {n["id"]: n for n in nodes}
    reachable: set[int] = set()
    stack: List[int] = list(outputs)
    while stack:
        nid = stack.pop()
        if nid not in node_map or nid in reachable:
            continue
        reachable.add(nid)
        stack.extend(node_map[nid].get("inputs", []))
    new_nodes = [node_map[nid] for nid in sorted(reachable) if nid in node_map]
    new_outputs = sorted([o for o in outputs if o in reachable])
    return new_nodes, new_outputs

def grade(sol, fx) -> dict:
    """Grade the candidate implementation against an oracle."""
    cases: List[Tuple[List[Dict[str, Any]], List[int]]] = [
        # simple chain
        (
            [
                {"id": 0, "op": "const", "inputs": []},
                {"id": 1, "op": "add",   "inputs": [0]},
                {"id": 2, "op": "mul",   "inputs": [1]}
            ],
            [2]
        ),
        # branch with dead node
        (
            [
                {"id": 0, "op": "const", "inputs": []},
                {"id": 1, "op": "add",   "inputs": [0]},
                {"id": 2, "op": "sub",   "inputs": [0]},
                {"id": 3, "op": "mul",   "inputs": [1, 2]}
            ],
            [3]
        ),
        # multiple outputs with one dead
        (
            [
                {"id": 0, "op": "const", "inputs": []},
                {"id": 1, "op": "add",   "inputs": [0]},
                {"id": 2, "op": "noop",  "inputs": []}
            ],
            [1, 2]
        ),
        # unreachable node not referenced anywhere
        (
            [
                {"id": 0, "op": "const", "inputs": []},
                {"id": 1, "op": "add",   "inputs": [0]},
                {"id": 2, "op": "mul",   "inputs": [3]},  # refers to non-existent id 3
                {"id": 3, "op": "noop",  "inputs": []}
            ],
            [1]
        ),
    ]

    ok = 1.0
    for nodes, outputs in cases:
        try:
            got_nodes, got_outputs = sol.dead_code_elimination(nodes, outputs)
        except Exception:
            return {"exact_match": 0.0}

        ref_nodes, ref_outputs = _reference(nodes, outputs)

        if got_nodes != ref_nodes or got_outputs != ref_outputs:
            ok = 0.0
            break

    return {"exact_match": ok}
