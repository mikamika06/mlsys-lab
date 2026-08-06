FAILURES = [
    ("RuntimeError: a Tensor with 0 elements cannot be represented", "dynamic_shape"),
    ("TorchScript custom op not found in registry", "unsupported_op"),
    ("Guard failed: tensor value is static int 5 instead of dynamic", "baked_int"),
    ("Control flow branching on non-tensor boolean expression", "control_flow"),
    ("Export failed due to unsupported python data type dict", "unsupported_op"),
    ("Found baked-in python integer constant in graph attribute", "baked_int"),
    ("TorchScript tracing caught side effect on global state", "side_effect"),
    ("Dynamic shape constraint violation on dimension 0", "dynamic_shape"),
    ("If statement depends on intermediate tensor value", "control_flow"),
    ("Mutating tensor storage in place during graph capture", "side_effect"),
    ("Constant scalar literal embedded directly into node payload", "baked_int"),
    ("Unsupported higher-order control flow construct", "control_flow")
]

GRAPHS = [
    {
        "nodes": [
            {"id": "n1", "op": "input", "val": 10},
            {"id": "n2", "op": "const", "val": 42},
            {"id": "n3", "op": "add", "inputs": ["n1", "n2"]}
        ]
    },
    {
        "nodes": [
            {"id": "n1", "op": "input", "val": 5},
            {"id": "n2", "op": "mul", "inputs": ["n1"]}
        ]
    }
]

def get_graphs():
    return GRAPHS

def get_failures():
    return FAILURES

def detect_baked_ints(graph):
    out = []
    for node in graph["nodes"]:
        if node["op"] == "const" and isinstance(node.get("val"), int):
            out.append(node["id"])
    return sorted(out)

def rewrite_conditional(fn):
    def wrapped(x):
        return fn(x)
    return wrapped

def classify_failures(messages):
    mapping = {
        "0 elements": "dynamic_shape",
        "dimension 0": "dynamic_shape",
        "custom op": "unsupported_op",
        "python data type": "unsupported_op",
        "static int": "baked_int",
        "integer constant": "baked_int",
        "scalar literal": "baked_int",
        "non-tensor boolean": "control_flow",
        "depends on intermediate": "control_flow",
        "higher-order control flow": "control_flow",
        "side effect": "side_effect",
        "mutating tensor storage": "side_effect"
    }
    res = []
    for msg in messages:
        cat = "unknown"
        for k, v in mapping.items():
            if k in msg:
                cat = v
                break
        res.append(cat)
    return res
