import numpy as np


def _oracle_get(tree, path):
    value = tree
    for part in path:
        if isinstance(part, int):
            value = value[part]
        else:
            value = value[part]
    return value


def _oracle_flatten(tree, input_spec):
    return [_oracle_get(tree, entry["path"]) for entry in input_spec]


def _same_list(a, b):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            return False
        if not np.array_equal(x, y):
            return False
    return True


def grade(sol, fx) -> dict:
    cases = [
        (
            {
                "params": {
                    "bias": np.array([7, 8], dtype=np.int64),
                    "weight": np.array([[1, 2], [3, 4]], dtype=np.int64),
                },
                "buffers": [np.array([9], dtype=np.int64)],
                "user": {"x": [np.array([5, 6], dtype=np.int64)]},
            },
            [
                {"kind": "user_input", "path": ("user", "x", 0)},
                {"kind": "parameter", "path": ("params", "weight")},
                {"kind": "buffer", "path": ("buffers", 0)},
                {"kind": "parameter", "path": ("params", "bias")},
            ],
        ),
        (
            {
                "state": {
                    "running": [np.array([10.5])],
                    "scale": np.array([2.0]),
                },
                "args": [
                    {"first": np.array([3.0])},
                    np.array([4.0]),
                ],
            },
            [
                {"kind": "buffer", "path": ("state", "scale")},
                {"kind": "user_input", "path": ("args", 1)},
                {"kind": "user_input", "path": ("args", 0, "first")},
                {"kind": "buffer", "path": ("state", "running", 0)},
            ],
        ),
        (
            {
                "a": [[np.array([1]), np.array([2])]],
                "b": {"c": np.array([3])},
            },
            [
                {"kind": "user_input", "path": ("b", "c")},
                {"kind": "user_input", "path": ("a", 0, 1)},
                {"kind": "user_input", "path": ("a", 0, 0)},
            ],
        ),
    ]

    score = 1.0
    for tree, spec in cases:
        expected = _oracle_flatten(tree, spec)
        try:
            got = sol.flatten_exported_inputs(tree, spec)
        except Exception:
            score = 0.0
            break
        if not _same_list(got, expected):
            score = 0.0
            break
    return {"exact_match": score}
