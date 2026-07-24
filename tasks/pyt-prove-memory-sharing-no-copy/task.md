## Context

Python's buffer protocol allows writable objects such as `bytearray` to expose
their underlying storage without copying. A `memoryview` provides access to that
storage while keeping the original memory as the owner of the bytes.

If two views share the same buffer $B$, a mutation through one view changes the
values observed through the other view. For two views $v_1$ and $v_2$ over the
same buffer, after writing

$$
B[i] \leftarrow x ,
$$

both views should observe

$$
v_1[i] = v_2[i] = x .
$$

A copied buffer does not have this property because the two objects have
different underlying storage.

## Task

Implement `prove_memory_sharing(values, index, new_value)`:

```python
def prove_memory_sharing(values: list[int], index: int, new_value: int) -> tuple[int, int, int]:
    ...
```

The function must:

1. Create one writable `bytearray` buffer from `values`.
2. Create two `memoryview` objects over that same buffer.
3. Record the value at `index` through the second view before any mutation.
4. Mutate the byte at `index` through the first view.
5. Return a tuple containing:
   - the value observed through the second view before the mutation,
   - the value observed through the first view after the mutation,
   - the value observed through the second view after the mutation.

The returned values must be read from the views, not constructed from the input
arguments.

## Example

```python
result = prove_memory_sharing([10, 20, 30], 1, 99)
# result == (20, 99, 99)
```

## What the gate checks

The gate creates an oracle using Python's real `bytearray` and `memoryview`
buffer behavior. It performs the mutation through one view and reads the values
through both views, then compares the returned tuple with the oracle result.

A solution passes only if it matches the observable behavior of two views that
share one writable buffer.
