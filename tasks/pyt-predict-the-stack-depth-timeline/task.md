## Context

A Python code object stores bytecode instructions that are executed by the evaluation loop. Many instructions consume values from the value stack and push new values back. The stack depth changes over time as each instruction executes.

For an instruction sequence $I_1, I_2, \dots, I_n$, let $s_k$ be the stack depth after executing instruction $I_k$. If an instruction has stack effect $\Delta_k$, then

$$
s_k = s_{k-1} + \Delta_k ,
$$

where $s_0 = 0$.

CPython exposes the stack effect of bytecode instructions through the `dis` module. The function `dis.stack_effect()` computes how an opcode changes the value stack, including instructions whose effect depends on an argument.

## Task

Implement `stack_depth_timeline(code)`:

```python
def stack_depth_timeline(code):
    ...
```

The argument is a Python code object. Return a list of integers where each element is the value stack depth immediately after the corresponding bytecode instruction in `dis.get_instructions(code)` order.

Use the instruction stream and opcode stack effects. The returned list must contain one entry per instruction and must not include the initial depth before execution.

## Example

```python
def add_one(x):
    return x + 1

timeline = stack_depth_timeline(add_one.__code__)

# The values depend on the running Python version.
# The function returns one integer per disassembled instruction.
```

## What the gate checks

The gate creates real Python code objects, disassembles them with `dis.get_instructions()`, and computes the expected timeline using CPython's own `dis.stack_effect()` implementation. Your function must return exactly the same list of stack depths.
