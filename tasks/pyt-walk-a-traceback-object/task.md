## Context

When Python raises an exception, the exception object stores a traceback object in
`exc.__traceback__`. A traceback is a linked structure. Each node contains a frame,
a source line number, and a reference to the next traceback node through
`tb_next`.

Walking from the first node to the end produces the call chain that led to the
exception. If the traceback nodes are $t_0, t_1, \dots, t_k$, the chain is the
ordered sequence

$$
[(f_0, \ell_0, n_0), (f_1, \ell_1, n_1), \dots, (f_k, \ell_k, n_k)]
$$

where $f_i$ is the source filename, $\ell_i$ is the line number, and $n_i$ is
the function name stored in the frame.

This task uses only the traceback object itself. The implementation should not
parse exception strings or inspect printed tracebacks.

## Task

Implement `walk_traceback(exc)`:

```python
def walk_traceback(exc):
    ...
```

The function receives a caught Python exception object and returns a list of
tuples. Each tuple must have the form:

```python
(basename, lineno, funcname)
```

where:

- `basename` is the final path component of the traceback frame filename.
- `lineno` is the traceback line number.
- `funcname` is the function name from the frame code object.

Walk the traceback using `exc.__traceback__` and `tb_next` until the chain ends.
The first tuple must correspond to the first traceback node.

## Example

```python
def inner():
    raise ValueError("bad")

try:
    inner()
except ValueError as exc:
    chain = walk_traceback(exc)
```

The returned list contains entries such as:

```python
[
    ("example.py", 6, "<module>"),
    ("example.py", 2, "inner")
]
```

The exact line numbers depend on where the code is placed.

## What the gate checks

The gate creates real Python exceptions, obtains the expected traceback chain by
walking CPython traceback objects directly, and compares it with the submitted
implementation.

The `exact_match` score must be $1.0$. Implementations that use formatted
traceback text, reverse the order, stop early, or return only the exception
location will not match the real traceback object chain.
