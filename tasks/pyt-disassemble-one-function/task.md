## Context

Python functions are compiled into code objects before execution. The `dis` module exposes the bytecode instructions stored in these objects.

For a function $f$, the instruction stream can be viewed as a sequence

$$I(f) = (i_0, i_1, \dots, i_k),$$

where each instruction has fields such as an opcode name, argument, and source location. The opcode name describes the operation performed by the virtual machine, such as loading a constant or returning a value.

The `dis.get_instructions` function provides a stable way to inspect this stream. Extracting only the opcode names produces a compact representation of the function's bytecode behavior.

## Task

Implement `disassemble_one_function(fn)`:

```python
def disassemble_one_function(fn):
    ...
```

The function receives a Python function object and must return a list of strings containing the `opname` value of every instruction produced by `dis.get_instructions(fn)`.

Do not execute the function. Only inspect its bytecode.

## Example

```python
def add_one(x):
    return x + 1

result = disassemble_one_function(add_one)

# Example output on the pinned Python version:
# ["RESUME", "LOAD_FAST", "LOAD_CONST", "BINARY_OP", "RETURN_VALUE"]
```

The exact instruction sequence depends on the Python version, so the expected values are obtained from the runtime disassembler.

## What the gate checks

The gate creates a fixture function and computes the expected opcode sequence using the real CPython `dis.get_instructions` implementation.

Your returned list must exactly match the oracle sequence. The comparison metric is `exact_match`, which must be $1.0$ to pass.
