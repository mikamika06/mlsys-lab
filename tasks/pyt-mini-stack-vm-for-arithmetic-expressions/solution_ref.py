def eval_vm(code):
    stack = []
    for op, arg in code:
        if op == "LOAD_CONST":
            stack.append(float(arg))
        elif op == "BINARY_OP":
            right = stack.pop()
            left = stack.pop()
            if arg == "+":
                stack.append(left + right)
            elif arg == "-":
                stack.append(left - right)
            elif arg == "*":
                stack.append(left * right)
            elif arg == "/":
                stack.append(left / right)
            else:
                raise ValueError("unknown binary operator")
        elif op == "UNARY_OP":
            value = stack.pop()
            if arg == "neg":
                stack.append(-value)
            elif arg == "abs":
                stack.append(abs(value))
            else:
                raise ValueError("unknown unary operator")
        elif op == "RETURN":
            return stack.pop()
        else:
            raise ValueError("unknown opcode")
    raise ValueError("missing return")
