def run_vm(program):
    stack = []
    for instruction in program:
        if instruction[0] == "PUSH_CONST":
            stack.append(instruction[1])
        elif instruction[0] == "BINARY_OP":
            op = instruction[1]
            right = stack.pop()
            left = stack.pop()
            if op == "+":
                stack.append(left + right)
            elif op == "-":
                stack.append(left - right)
            elif op == "*":
                stack.append(left * right)
            elif op == "/":
                stack.append(left / right)
            elif op == "//":
                stack.append(left // right)
            elif op == "%":
                stack.append(left % right)
    return stack[-1]
