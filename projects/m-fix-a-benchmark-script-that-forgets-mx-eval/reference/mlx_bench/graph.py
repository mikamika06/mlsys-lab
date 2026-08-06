import mlx.core as mx


def analyze_implicit_evals(operations):
    triggers = []
    for i, op in enumerate(operations):
        if op in ("item", "tolist", "numpy", "print", "bool", "float", "int"):
            triggers.append(i)
    return triggers
