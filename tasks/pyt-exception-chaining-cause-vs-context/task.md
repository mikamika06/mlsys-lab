## Context

Python records relationships between exceptions when one exception is raised while another
exception is being handled. There are two different mechanisms.

An explicit chain created with `raise ... from ...` stores the original exception in
`__cause__`. An implicit chain created by raising a new exception inside an `except`
block without `from` stores the previous exception in `__context__`.

For an exception object $e$, the two relationships can be viewed as optional references:

$$
e.\_\_cause\_\_ \in \{\text{exception object}, \mathrm{None}\}
$$

and

$$
e.\_\_context\_\_ \in \{\text{exception object}, \mathrm{None}\}.
$$

The type names of these referenced exceptions describe the visible chain.

## Task

Implement `inspect_exception_chain(mode)`.

The function receives one of these strings:

- `"from"`: create an exception where an inner `ValueError` is converted into a
  `RuntimeError` using `raise RuntimeError(...) from ...`.
- `"context"`: create an exception where a `ValueError` is handled and a
  `RuntimeError` is raised without `from`.

Return a dictionary with exactly these keys:

```python
{
    "cause": bool,
    "context": bool,
    "cause_type": str | None,
    "context_type": str | None
}
```

The boolean fields indicate whether `__cause__` or `__context__` is not `None`.
The type fields contain the class name of the referenced exception, or `None`
when the reference is absent.

The function should inspect the exception objects rather than returning fixed
answers for the two input strings.

## Example

```python
print(inspect_exception_chain("from"))
# {
#   "cause": True,
#   "context": True,
#   "cause_type": "ValueError",
#   "context_type": "ValueError"
# }

print(inspect_exception_chain("context"))
# {
#   "cause": False,
#   "context": True,
#   "cause_type": None,
#   "context_type": "ValueError"
# }
```

## What the gate checks

The gate creates the expected results by running the same exception chaining
operations on the real CPython interpreter and inspecting the resulting
exception objects.

Your implementation must return the same cause/context presence flags and
exception type names for both chaining modes. Hardcoding the visible examples
does not satisfy the intended contract when the implementation logic is
different from Python's exception metadata behavior.
