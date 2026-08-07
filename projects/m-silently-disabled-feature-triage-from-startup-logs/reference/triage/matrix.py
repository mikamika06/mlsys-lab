import ast

def extract_support_matrix(source_code):
    tree = ast.parse(source_code)
    matrix = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = []
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef):
                    methods.append(sub.name)
            matrix[node.name] = sorted(methods)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    if isinstance(node.value, ast.Constant):
                        matrix[target.id] = node.value.value
    return matrix
