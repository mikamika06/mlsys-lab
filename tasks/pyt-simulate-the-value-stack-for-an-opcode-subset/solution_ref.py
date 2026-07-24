def simulate_value_stack(instructions, constants, variables):
    stack = []
    for opcode, arg in instructions:
        if opcode == "LOAD_CONST":
            stack.append(constants[arg])
        elif opcode == "LOAD_NAME":
            stack.append(variables[arg])
        elif opcode == "BINARY_ADD":
            b = stack.pop()
            a = stack.pop()
            stack.append(a + b)
        elif opcode == "BINARY_MULTIPLY":
            b = stack.pop()
            a = stack.pop()
            stack.append(a * b)
        elif opcode == "UNARY_NEGATIVE":
            stack.append(-stack.pop())
        elif opcode == "BUILD_TUPLE":
            values = stack[-arg:]
            del stack[-arg:]
            stack.append(tuple(values))
        elif opcode == "RETURN_VALUE":
            return stack[-1]
    raise RuntimeError("missing RETURN_VALUE")
