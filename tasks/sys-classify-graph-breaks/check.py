import ast

def _is_traceable(code_str):
    try:
        tree = ast.parse(code_str, mode='eval')
    except SyntaxError:
        # If eval fails, try function definition mode
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

def grade(sol, fx) -> dict:
    snippets = [
        "lambda x: x + 1",
        "lambda x: x * y",
        "def f(x): return x + 2",
        "def g(x):\\n    if x > 0:\\n        return x\\n    else:\\n        return -x",
        "lambda a, b: a / b",
        "def h(x):\\n    for i in range(10):\\n        x += i",
        "lambda x: (x + 1) * (x - 2)",
        "def k(x):\\n    return sum([i*i for i in range(x)])",
        "lambda x: __import__('math').sqrt(x)",
        "def l(x):\\n    while x > 0:\\n        x -= 1"
    ]
    expected = ["traceable" if _is_traceable(s) else "break" for s in snippets]
    try:
        got = sol.classify_breaks(snippets)
    except Exception:
        return {"exact_match": 0.0}
    if not isinstance(got, list) or len(got) != len(snippets):
        return {"exact_match": 0.0}
    ok = all(a == b for a, b in zip(got, expected))
    return {"exact_match": float(ok)}
