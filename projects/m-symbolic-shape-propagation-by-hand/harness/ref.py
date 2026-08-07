import symshape.infer as infer
import symshape.coverage as coverage


GRAPHS = [
    {
        "inputs": {
            "x": ("B", "S", 64),
            "w": (64, 128)
        },
        "nodes": [
            {
                "name": "mm",
                "op": "MatMul",
                "inputs": ["x", "w"],
                "params": {}
            },
            {
                "name": "bias",
                "op": "Add",
                "inputs": ["mm", "w"],
                "params": {}
            }
        ]
    },
    {
        "inputs": {
            "a": ("B", 128),
            "b": ("B", 256)
        },
        "nodes": [
            {
                "name": "cat",
                "op": "Concat",
                "inputs": ["a", "b"],
                "params": {"axis": 1}
            },
            {
                "name": "resh",
                "op": "Reshape",
                "inputs": ["cat"],
                "params": {"shape": ["B", -1]}
            }
        ]
    },
    {
        "inputs": {
            "p": ("B", 16, 32),
            "q": ("B", 8, 32)
        },
        "nodes": [
            {
                "name": "bad_add",
                "op": "Add",
                "inputs": ["p", "q"],
                "params": {}
            },
            {
                "name": "next_node",
                "op": "Transpose",
                "inputs": ["bad_add"],
                "params": {"perm": [0, 2, 1]}
            }
        ]
    }
]


def propagate_shapes(graph):
    return infer.propagate_shapes(graph)


def find_first_failure(graph):
    return infer.find_first_failure(graph)


def compute_coverage(graph):
    return coverage.compute_coverage(graph)
