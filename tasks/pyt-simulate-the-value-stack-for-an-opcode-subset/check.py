def _to_python_expression(instructions):
    stack = []
    for opcode, arg in instructions:
        if opcode == "LOAD_CONST":
            stack.append(f"constants[{arg}]")
        elif opcode == "LOAD_NAME":
            stack.append(f"variables[{arg!r}]")
        elif opcode == "BINARY_ADD":
            b = stack.pop()
            a = stack.pop()
            stack.append(f"({a} + {b})")
        elif opcode == "BINARY_MULTIPLY":
            b = stack.pop()
            a = stack.pop()
            stack.append(f"({a} * {b})")
        elif opcode == "UNARY_NEGATIVE":
            a = stack.pop()
            stack.append(f"(-({a}))")
        elif opcode == "BUILD_TUPLE":
            vals = stack[-arg:]
            del stack[-arg:]
            stack.append("(" + ", ".join(vals) + ("," if arg == 1 else "") + ")")
        elif opcode == "RETURN_VALUE":
            return stack[-1]
    raise RuntimeError("missing return")


def _oracle(instructions, constants, variables):
    expr = _to_python_expression(instructions)
    return eval(expr, {"__builtins__": {}}, {"constants": constants, "variables": variables})


def grade(sol, fx) -> dict:
    cases = [
        (
            [
                ("LOAD_CONST", 0),
                ("LOAD_NAME", "x"),
                ("BINARY_ADD", None),
                ("RETURN_VALUE", None),
            ],
            [7],
            {"x": 5},
        ),
        (
            [
                ("LOAD_NAME", "a"),
                ("LOAD_NAME", "b"),
                ("BINARY_MULTIPLY", None),
                ("UNARY_NEGATIVE", None),
                ("RETURN_VALUE", None),
            ],
            [],
            {"a": 6, "b": 4},
        ),
        (
            [
                ("LOAD_CONST", 0),
                ("LOAD_CONST", 1),
                ("LOAD_NAME", "x"),
                ("BINARY_ADD", None),
                ("BUILD_TUPLE", 2),
                ("RETURN_VALUE", None),
            ],
            [3, 9],
            {"x": 2},
        ),
        (
            [
                ("LOAD_NAME", "x"),
                ("UNARY_NEGATIVE", None),
                ("LOAD_CONST", 0),
                ("BINARY_MULTIPLY", None),
                ("RETURN_VALUE", None),
            ],
            [8],
            {"x": 5},
        ),
    ]

    ok = 1.0
    for instructions, constants, variables in cases:
        try:
            expected = _oracle(instructions, constants, variables)
            got = sol.simulate_value_stack(
                list(instructions),
                list(constants),
                dict(variables),
            )
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
