import ast

def _is_traceable(code_str):
    try:
        tree = ast.parse(code_str, mode='eval')
    except SyntaxError:
        try:
            tree = ast.parse(code_str, mode='exec')
        except Exception:
            return False
    allowed_nodes = {
        ast.Expression, ast.Lambda, ast.BinOp, ast.UnaryOp,
        ast.Constant, ast.Name, ast.Load, ast.Add, ast.Sub,
        ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
        ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd,
        ast.UAdd, ast.USub
    }
    for node in ast.walk(tree):
        if type(node) not in allowed_nodes:
            return False
    return True

def classify_breaks(snippets: list[str]) -> list[str]:
    """
    Classify each code snippet as 'traceable' or 'break'.
    A snippet is traceable iff its AST contains only nodes from the allowed set.
    """
    result = []
    for s in snippets:
        if _is_traceable(s):
            result.append("traceable")
        else:
            result.append("break")
    return result
