def run_vm(program):
    # TODO: broken stack effect. It pops the left operand first, reversing
    # subtraction, division, floor division, and modulo operations.
    stack = []
    for instruction in program:
        if instruction[0] == "PUSH_CONST":
            stack.append(instruction[1])
        elif instruction[0] == "BINARY_OP":
            op = instruction[1]
            left = stack.pop()
            right = stack.pop()
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
