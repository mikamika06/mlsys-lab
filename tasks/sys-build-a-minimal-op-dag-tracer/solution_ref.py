import dis


def trace_function(fn):
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
            op = {
                "BINARY_ADD": "add",
                "BINARY_SUBTRACT": "sub",
                "BINARY_MULTIPLY": "mul",
                "BINARY_TRUE_DIVIDE": "truediv",
            }.get(ins.opname)
            if op is not None:
                out = "t" + str(counter)
                counter += 1
                nodes.append((op, (left, right), out))
                stack.append(out)

    return nodes
