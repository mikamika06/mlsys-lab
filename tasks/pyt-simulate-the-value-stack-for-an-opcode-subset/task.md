## Context

A Python evaluation loop executes instructions while maintaining a value stack. Each opcode consumes values from the stack and may push new values back.

For a stack $S$, a binary operation removes the top two operands and pushes the result:

$$
S \leftarrow S[:-2] + [S[-2] \mathbin{\mathrm{op}} S[-1]] .
$$

A stack machine separates the representation of a computation from the execution of that computation. The instruction sequence determines how values move through the stack.

In this task, the supported instruction subset models a small part of CPython's evaluation model. The simulator does not execute arbitrary Python bytecode. It only needs to implement the listed stack effects.

Supported opcodes:

- `LOAD_CONST i`: push `constants[i]`.
- `LOAD_NAME name`: push `variables[name]`.
- `BINARY_ADD`: pop two values and push their sum.
- `BINARY_MULTIPLY`: pop two values and push their product.
- `UNARY_NEGATIVE`: replace the top value with its negation.
- `BUILD_TUPLE n`: pop `n` values and push a tuple in original stack order.
- `RETURN_VALUE`: return the top value.

## Task

Implement `simulate_value_stack(instructions, constants, variables)`.

The arguments are:

- `instructions`: a list of `(opcode, argument)` tuples. Instructions without an argument use `None`.
- `constants`: a list of constant values.
- `variables`: a dictionary containing names used by `LOAD_NAME`.

Return the value produced by the final `RETURN_VALUE` instruction.

The input programs are guaranteed to be valid for this opcode subset. Use only the stack behavior described above.

## Example

```python
instructions = [
    ("LOAD_CONST", 2),
    ("LOAD_NAME", "x"),
    ("BINARY_ADD", None),
    ("UNARY_NEGATIVE", None),
    ("RETURN_VALUE", None),
]

constants = [10, 20, 30]
variables = {"x": 5}

# The stack evolves as:
# [30] -> [30, 5] -> [35] -> [-35]
```

The result is:

```python
-35
```

## What the gate checks

The gate generates several valid programs from this opcode subset. For each program, it creates an equivalent Python expression and evaluates it with CPython's `eval` implementation.

The returned value from `simulate_value_stack` must exactly match the CPython result. The gate reports `exact_match`, which must equal $1.0$ for all cases.
