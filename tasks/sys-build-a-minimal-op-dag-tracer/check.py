import dis


def _oracle_trace(fn):
    stack = []
    nodes = []
    counter = 0

    binary = {
        "+": "add",
        "-": "sub",
        "*": "mul",
        "/": "truediv",
    }

    for ins in dis.get_instructions(fn):
        if ins.opname == "LOAD_FAST":
            stack.append(ins.argval)
        elif ins.opname == "LOAD_CONST":
            stack.append("const:" + str(ins.argval))
        elif ins.opname == "BINARY_OP":
            right = stack.pop()
            left = stack.pop()
            op = binary.get(ins.argrepr)
            if op is not None:
                out = "t" + str(counter)
                counter += 1
                nodes.append((op, (left, right), out))
                stack.append(out)
        elif ins.opname.startswith("BINARY_"):
            right = stack.pop()
            left = stack.pop()
            name = {
                "BINARY_ADD": "add",
                "BINARY_SUBTRACT": "sub",
                "BINARY_MULTIPLY": "mul",
                "BINARY_TRUE_DIVIDE": "truediv",
            }.get(ins.opname)
            if name is not None:
                out = "t" + str(counter)
                counter += 1
                nodes.append((name, (left, right), out))
                stack.append(out)

    return nodes


def grade(sol, fx) -> dict:
    def f1(x, y):
        return (x + y) * 2

    def f2(a, b, c):
        return a * b + c

    def f3(x, y):
        return (x - 3) / y

    cases = [f1, f2, f3]

    ok = 1.0
    for fn in cases:
        try:
            got = sol.trace_function(fn)
            got = [
                (op, tuple(args), out)
                for op, args, out in got
            ]
            ref = _oracle_trace(fn)
        except Exception:
            ok = 0.0
            break
        if got != ref:
            ok = 0.0
            break

    return {"exact_match": ok}
