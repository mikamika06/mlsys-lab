## Context

A Python function is represented internally by a code object. One important field
of a code object is `co_code`, the byte sequence containing the virtual machine
instructions executed by CPython.

For a small function, the bytecode sequence can be inspected with `dis` and
recreated by emitting the same opcode bytes. The exact bytes depend on the
CPython version because the virtual machine instruction set changes between
releases.

The target function is:

```python
def add_two(a, b):
    return a + b
```

The bytecode must match the `co_code` bytes produced by the real CPython 3.12
compiler for this function.

## Task

Implement:

```python
def assemble_add_two_bytecode() -> bytes:
    ...
```

Return only the `co_code` byte sequence of the compiled `add_two` function.

Do not return a code object, a function object, or a disassembly string.

## Example

```python
code = assemble_add_two_bytecode()

# code is the exact bytes stored in:
# add_two.__code__.co_code
```

## What the gate checks

The gate compiles the reference function using the running CPython interpreter
and reads its real `co_code` field. It compares the returned bytes with this
oracle using `byte_exact_fraction`.

A score of $1.0$ is required, meaning every byte must match exactly. The gate
therefore checks both opcode selection and operand encoding.
