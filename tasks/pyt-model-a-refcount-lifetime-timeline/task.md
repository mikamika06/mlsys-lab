## Context

CPython uses reference counting to track how many live references point to an
object. When the count reaches zero, the object can be reclaimed. If an object
has $r$ live references, each new owning reference changes the count by one:

$$
r_{\mathrm{new}} = r_{\mathrm{old}} + \Delta r .
$$

Different Python actions affect the count differently. Assigning another name,
creating an alias, and inserting an object into a container create persistent
references. Passing an object as a function argument creates only a temporary
reference that disappears after the call. Returning an object creates a new
persistent reference when the returned value is stored.

This task models a sequence of operations and reports the object's reference
count after each operation.

## Task

Implement `refcount_timeline(steps)`:

```python
def refcount_timeline(steps: list[str]) -> list[int]:
    ...
```

The input is a list of operation names. Start with one live reference to an
object. Process each operation in order and return the reference count after
each operation.

Supported operations:

- `"assign"` creates one new name binding to the object.
- `"alias"` creates another alias to the same object.
- `"container-insert"` stores the object in a container.
- `"function-arg"` passes the object to a function. The temporary argument
  reference disappears before the next operation.
- `"return"` stores a returned reference to the object.

The output list must contain one integer per input operation.

## Example

```python
steps = [
    "assign",
    "alias",
    "container-insert",
    "function-arg",
    "return",
]

print(refcount_timeline(steps))
# [2, 3, 4, 4, 5]
```

## What the gate checks

The gate runs the implementation on several operation sequences. The expected
timeline is computed from a live CPython object using `sys.getrefcount` while
performing the same reference operations. The returned list must exactly match
the CPython-derived timeline.
