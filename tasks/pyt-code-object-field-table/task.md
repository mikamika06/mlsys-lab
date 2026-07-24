## Context

Every Python function carries a compiled **code object** in `fn.__code__`.
It's not just bytecode — it's a small struct of metadata CPython fills in at
compile time, describing the shape of the function's calling convention and
the frame CPython must allocate to run it:

* `co_argcount` — number of positional parameters (not counting `*args`,
  `**kwargs`, or keyword-only parameters).
* `co_nlocals` — number of local variable slots the frame needs: parameters
  plus every locally-assigned name in the function body.
* `co_stacksize` — the maximum depth CPython's bytecode value stack reaches
  while executing this function, computed once at compile time so the frame
  can pre-allocate exactly that much space.
* `co_flags` — a bitmask of properties such as whether the function uses
  `*args` (`CO_VARARGS = 0x04`), `**kwargs` (`CO_VARKEYWORDS = 0x08`), or is
  a generator (`CO_GENERATOR = 0x20`).
* `co_consts` — the tuple of literal constants embedded in the function
  (including its docstring, if any, and nested code objects for any
  functions/comprehensions defined inside it).
* `co_names` — the tuple of non-local names the function looks up by name at
  runtime (globals, attributes, imported names — not local variables, which
  live in `co_nlocals` slots instead).

## Task

Implement `code_fields`:

```python
def code_fields(fn):
    """Return (co_argcount, co_nlocals, co_stacksize, co_flags,
    len(co_consts), len(co_names)) for fn's code object."""
```

`fn` is any Python callable that has a `__code__` attribute (a plain
function, a closure, a generator function, or a bound/unbound method).
Read the six fields directly off `fn.__code__` and return them as a 6-tuple
in the order listed above — the last two entries are *lengths* of the
`co_consts` and `co_names` tuples, not the tuples themselves.

## Example

```python
def f(a, b, c=3):
    x = a + b + c
    y = x * 2
    return y

code_fields(f)
# -> (3, 5, 2, 3, 2, 0)
#     co_argcount=3 (a, b, c)
#     co_nlocals=5  (a, b, c, x, y)
#     co_stacksize=2
#     co_flags=3    (CO_OPTIMIZED | CO_NEWLOCALS, no *args/**kwargs/generator bits)
#     len(co_consts)=2  (None, and the literal 2)
#     len(co_names)=0   (nothing looked up by name — everything is local)
```

## What the gate checks

The grader defines five fixture callables covering different shapes: a
plain function with a default argument, a `*args`/`**kwargs` function, a
generator function, a closure returning a nested function, and a bound
method. For each one it reads the six fields straight off the live
`fn.__code__` object (there is no separate formula to re-derive — this
introspection *is* the ground truth) and compares that 6-tuple to what your
`code_fields` returns. **exact_match** is `1.0` only if every fixture's
tuple matches exactly; any mismatch, wrong order, or exception drops it to
`0.0`.
