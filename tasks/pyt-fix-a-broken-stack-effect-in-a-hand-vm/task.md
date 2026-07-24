## Context

A stack-based virtual machine evaluates instructions by consuming values from a
stack and pushing results back. A binary operation has the stack effect

$$
[a, b] \rightarrow [f(a,b)] .
$$

The top of the stack is the right operand. Therefore, for an instruction such as
subtraction, a VM must pop the operands in this order:

$$
b = \operatorname{pop}(), \qquad a = \operatorname{pop}(), \qquad
\operatorname{push}(a-b).
$$

Swapping these pops changes the meaning of non-commutative operations. For
example, $8-3$ is not equal to $3-8$.

This task uses a small hand-written VM. A real Python expression evaluator is
used as the oracle during grading.

## Task

Implement `run_vm(program)`.

The input is a list of instructions. Each instruction is a tuple:

- `("PUSH_CONST", value)` pushes a constant onto the value stack.
- `("BINARY_OP", op)` pops two values and pushes the result.

Supported binary operators are:

- `"+"`
- `"-"`
- `"*"`
- `"/"`
- `"//"`
- `"%"`

The function returns the final value on the stack after executing all
instructions.

The VM should apply binary operators using the normal left-to-right Python
meaning. For a stack containing `[left, right]`, `"BINARY_OP"` must compute
`left op right`.

## Example

```python
program = [
    ("PUSH_CONST", 10),
    ("PUSH_CONST", 3),
    ("BINARY_OP", "-"),
]

result = run_vm(program)
# 7
```

## What the gate checks

The gate generates several instruction programs and compares `run_vm` against a
reference result produced by evaluating the equivalent expression with the real
Python interpreter.

The `exact_match` metric must be $1.0$. A VM that reverses the operand pop order
will fail cases containing subtraction, division, floor division, or modulo.
