import ast

def verify_resume_signature(decompiled_code: str, expected_name: str) -> bool:
    try:
        tree = ast.parse(decompiled_code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == expected_name:
            return True
    return False
