## Context

NumPy arrays separate the logical shape of an array from the physical memory
layout. A view reuses the same underlying buffer and changes metadata such as
shape or strides. A copy allocates separate storage.

For two arrays $A$ and $B$, NumPy can test whether they overlap in memory with

$$
\mathrm{shares\_memory}(A,B).
$$

Operations that preserve a compatible stride layout can often return views.
Operations that require a different contiguous layout may need a copy. The
answer can depend on previous operations because an intermediate array may no
longer be contiguous.

## Task

Implement `predict_view_copy(ops)`:

```python
def predict_view_copy(ops: list[str]) -> list[str]:
    ...
```

The input is a sequence of NumPy operation names. Start from the base array

```python
np.arange(12, dtype=np.int64).reshape(3, 4)
```

and apply each operation in order. Return a list containing `"view"` or
`"copy"` for each operation. Each prediction must describe whether the output
of that operation shares memory with the array before the operation.

Supported operations:

- `"reshape_2x6"`: apply `reshape(2, 6)`.
- `"slice_step2"`: apply `array[::2]`.
- `"transpose"`: apply `array.T`.
- `"ravel"`: apply `array.ravel()`.

The function must classify the actual NumPy behavior. The sequence matters:
for example, `ravel` after `transpose` can behave differently from `ravel` on
the original contiguous array.

## Example

```python
ops = ["transpose", "ravel"]
result = predict_view_copy(ops)

# result:
# ["view", "copy"]
```

The transpose changes the strides of the array. Flattening that non-contiguous
layout requires a new contiguous buffer.

## What the gate checks

The gate executes the same operation sequences using NumPy and compares the
predictions with `np.shares_memory`.

The `exact_match` score must be exactly $1.0$. A classifier that assumes every
operation is a view or every flattening operation is a copy will fail on
sequences where the memory layout changes.
