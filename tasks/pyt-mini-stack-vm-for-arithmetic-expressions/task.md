## Context

A stack-based virtual machine evaluates arithmetic programs by keeping intermediate values in a value stack. Each instruction transforms the stack state.

A binary operation consumes two values. If the stack contains

$$
[\dots, a, b],
$$

then the operation uses $a$ as the left operand and $b$ as the right operand. This distinction is important because subtraction and division are not commutative:

$$
a - b \neq b - a
$$

and

$$
\frac{a}{b} \neq \frac{b}{a}.
$$

A VM separates program representation from execution. The bytecode describes the operations while the evaluation loop implements their meaning.

## Task

Implement `eval_vm(code)`:

```python
def eval_vm(code):
    ...
```

The input is a list of instruction tuples. Each tuple contains an opcode and an argument.

Supported instructions:

- `("LOAD_CONST", value)` pushes a numeric constant.
- `("BINARY_OP", op)` pops two values and applies `+`, `-`, `*`, or `/`.
- `("UNARY_OP", op)` pops one value and applies `neg` or `abs`.
- `("RETURN", None)` returns the final stack value.

The VM must execute the instructions using an explicit Python list as the stack. Do not use `eval`, `exec`, or parse source expressions.

## Example

```python
code = [
    ("LOAD_CONST", 3.0),
    ("LOAD_CONST", 4.0),
    ("BINARY_OP", "*"),
    ("UNARY_OP", "neg"),
    ("RETURN", None),
]

result = eval_vm(code)
# -12.0
```

## What the gate checks

The gate evaluates several bytecode programs. It computes each expected result with an independent arithmetic oracle and compares the VM output.

The relative error is

$$
\mathrm{rel\_err} =
\frac{|y_{\mathrm{vm}}-y_{\mathrm{ref}}|}
{|y_{\mathrm{ref}}|+10^{-12}}.
$$

The maximum relative error over all programs must satisfy

$$
\mathrm{rel\_err} \le 10^{-12}.
$$
