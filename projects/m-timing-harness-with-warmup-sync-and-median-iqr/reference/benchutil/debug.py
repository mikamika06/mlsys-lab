import ast


def find_methodology_bugs(code_str):
    tree = ast.parse(code_str)
    bugs = set()
    has_sync = False
    has_warmup = False
    uses_mean = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and "sync" in node.func.attr.lower():
                has_sync = True
            if isinstance(node.func, ast.Name) and "sync" in node.id.lower():
                has_sync = True
            if isinstance(node.func, ast.Name) and node.func.id in ("mean", "average"):
                uses_mean = True
            if isinstance(node.func, ast.Attribute) and node.func.attr in ("mean", "average"):
                uses_mean = True
            if isinstance(node.func, ast.Name) and "warmup" in node.func.id.lower():
                has_warmup = True
        if isinstance(node, ast.FunctionDef) and "warmup" in node.name.lower():
            has_warmup = True
    if not has_sync:
        bugs.add("missing_sync")
    if not has_warmup:
        bugs.add("missing_warmup")
    if uses_mean:
        bugs.add("uses_mean_instead_of_median")
    return sorted(list(bugs))
