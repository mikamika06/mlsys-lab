import ast
import operator


def _to_python_expr(program):
    stack = []
    for op in program:
        if op[0] == "PUSH_CONST":
            stack.append(repr(op[1]))
        elif op[0] == "BINARY_OP":
            right = stack.pop()
            left = stack.pop()
            stack.append(f"({left}{op[1]}{right})")
    return stack[-1]


def _oracle(program):
    expr = _to_python_expr(program)
    tree = ast.parse(expr, mode="eval")
    allowed = (
        ast.Expression,
        ast.BinOp,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError("unexpected expression node")
    return eval(expr, {"__builtins__": {}}, {})


def grade(sol, fx) -> dict:
    cases = [
        [
            ("PUSH_CONST", 10),
            ("PUSH_CONST", 3),
            ("BINARY_OP", "-"),
        ],
        [
            ("PUSH_CONST", 8),
            ("PUSH_CONST", 2),
            ("BINARY_OP", "/"),
        ],
        [
            ("PUSH_CONST", 17),
            ("PUSH_CONST", 5),
            ("BINARY_OP", "//"),
        ],
        [
            ("PUSH_CONST", 19),
            ("PUSH_CONST", 6),
            ("BINARY_OP", "%"),
        ],
        [
            ("PUSH_CONST", 4),
            ("PUSH_CONST", 7),
            ("BINARY_OP", "*"),
            ("PUSH_CONST", 3),
            ("BINARY_OP", "+"),
        ],
    ]
    ok = 1.0
    for program in cases:
        try:
            got = sol.run_vm(program)
            ref = _oracle(program)
        except Exception:
            ok = 0.0
            break
        if got != ref:
            ok = 0.0
            break
    return {"exact_match": ok}
