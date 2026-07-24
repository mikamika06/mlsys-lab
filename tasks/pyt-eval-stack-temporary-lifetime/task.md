## Context

Python evaluates expressions using an evaluation stack. Intermediate values are
temporaries held by references on that stack until an opcode consumes or drops
them.

Consider a temporary object created inside an expression. Its lifetime ends when
the bytecode instruction that consumes the owning temporary container releases
the last reference. Reference counting can therefore make the deallocation point
depend on the exact bytecode sequence.

The task uses the expression:

$$
(\mathrm{Temp}(), 0)[1]
$$

The tuple created for the subscription contains a temporary object. The
subscription operation consumes the tuple, and the tuple release decrements the
reference count of the first element. The bytecode instruction responsible for
that operation can be located from CPython's disassembly.

A bytecode step is defined as the zero-based ordinal position of an instruction
returned by `dis.get_instructions()` for the helper expression function.

## Task

Implement `temporary_lifetime_step()`:

```python
def temporary_lifetime_step() -> int:
    ...
```

The function must return the instruction index where the temporary object in the
expression `(\mathrm{Temp}(), 0)[1]` is released. Derive the answer from the
CPython bytecode structure rather than returning a guessed constant.

Use only the standard library.

## Example

```python
step = temporary_lifetime_step()
# step is the ordinal index of the BINARY_SUBSCR instruction
# in the helper expression bytecode.
```

## What the gate checks

The gate builds a real CPython bytecode oracle using `dis.get_instructions()`.
It locates the instruction that consumes the temporary tuple and compares the
returned step index against that oracle with exact equality.

A solution that assumes a fixed index without analyzing the current bytecode
layout may fail when the interpreter's generated instruction sequence changes.
