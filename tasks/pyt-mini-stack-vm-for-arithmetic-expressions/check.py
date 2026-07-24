import operator
import numpy as np


def _oracle(expr):
    kind = expr[0]
    if kind == "const":
        return float(expr[1])
    if kind == "unary":
        value = _oracle(expr[2])
        if expr[1] == "neg":
            return -value
        if expr[1] == "abs":
            return abs(value)
        raise ValueError("bad unary")
    if kind == "binary":
        left = _oracle(expr[2])
        right = _oracle(expr[3])
        ops = {
            "+": operator.add,
            "-": operator.sub,
            "*": operator.mul,
            "/": operator.truediv,
        }
        return float(ops[expr[1]](left, right))
    raise ValueError("bad expression")


def _programs():
    return [
        (
            [
                ("LOAD_CONST", 3.0),
                ("LOAD_CONST", 4.0),
                ("BINARY_OP", "*"),
                ("UNARY_OP", "neg"),
                ("RETURN", None),
            ],
            ("unary", "neg", ("binary", "*", ("const", 3.0), ("const", 4.0))),
        ),
        (
            [
                ("LOAD_CONST", 10.0),
                ("LOAD_CONST", 2.0),
                ("BINARY_OP", "/"),
                ("LOAD_CONST", 7.0),
                ("BINARY_OP", "+"),
                ("RETURN", None),
            ],
            ("binary", "+",
                ("binary", "/", ("const", 10.0), ("const", 2.0)),
                ("const", 7.0)),
        ),
        (
            [
                ("LOAD_CONST", -5.0),
                ("UNARY_OP", "abs"),
                ("LOAD_CONST", 8.0),
                ("BINARY_OP", "-"),
                ("RETURN", None),
            ],
            ("binary", "-",
                ("unary", "abs", ("const", -5.0)),
                ("const", 8.0)),
        ),
        (
            [
                ("LOAD_CONST", 1.5),
                ("LOAD_CONST", 2.0),
                ("BINARY_OP", "+"),
                ("LOAD_CONST", 3.0),
                ("BINARY_OP", "/"),
                ("LOAD_CONST", -4.0),
                ("BINARY_OP", "*"),
                ("RETURN", None),
            ],
            ("binary", "*",
                ("binary", "/",
                    ("binary", "+", ("const", 1.5), ("const", 2.0)),
                    ("const", 3.0)),
                ("const", -4.0)),
        ),
    ]


def grade(sol, fx) -> dict:
    errs = []
    for program, expr in _programs():
        try:
            got = float(sol.eval_vm(program))
            ref = _oracle(expr)
        except Exception:
            return {"rel_err": float("inf")}
        errs.append(abs(got - ref) / (abs(ref) + 1e-12))
    return {"rel_err": float(np.max(errs))}
