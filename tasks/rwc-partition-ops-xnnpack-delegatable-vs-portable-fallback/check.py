import sys
from typing import List, Tuple, Set, Dict

def _reference(nodes: List[Tuple[int, str]], support: Set[str]) -> Dict[int, bool]:
    """Oracle implementation of the partitioning rule."""
    return {node_id: op_name in support for node_id, op_name in nodes}

def grade(sol, fx) -> dict:
    # Define a handful of deterministic test cases
    tests = [
        (
            [(0, 'conv2d'), (1, 'relu'), (2, 'add')],
            {'conv2d', 'add'}
        ),
        (
            [],
            set()
        ),
        (
            [(10, 'matmul'), (20, 'softmax'), (30, 'tanh')],
            {'matmul', 'tanh'}
        ),
        (
            [(5, 'relu'), (6, 'relu'), (7, 'relu')],
            {'relu'}
        ),
        (
            [(i, name) for i, name in enumerate(['conv2d', 'add', 'softmax',
                                                  'matmul', 'relu'])],
            {'conv2d', 'softmax', 'tanh'}  # only two match
        )
    ]

    ok = 1.0
    try:
        for nodes, support in tests:
            got = sol.partition_ops(nodes, support)
            expected = _reference(nodes, support)
            if got != expected:
                ok = 0.0
                break
    except Exception as e:
        # Any exception is treated as a failure
        ok = 0.0

    return {"exact_match": ok}
