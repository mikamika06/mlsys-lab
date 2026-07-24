## Context

CPython uses reference counting to track how many references point to an object. The function `sys.getrefcount(x)` reports the current reference count, but the value includes an extra temporary reference created by passing `x` as an argument.

If a measurement helper returns

$$r(x) = \texttt{sys.getrefcount}(x) - 1,$$

the result removes that argument artifact. Adding or removing aliases changes the reference count by the number of new or deleted bindings.

For example, if a list receives another reference to an object, the count increases by one. If that list element is removed, the count decreases by one. The task uses a fixed sequence of bindings and asks for the deltas at checkpoints.

## Task

Implement:

```python
def predict_refcount_deltas() -> list[int]:
    ...
```

The function must execute the scripted binding sequence below and return the changes in reference count at each checkpoint relative to the initial checkpoint.

The sequence is:

1. Create an object `x`.
2. Record the initial reference count.
3. Create a list containing `x` and record a checkpoint.
4. Append another alias of `x` to the list and record a checkpoint.
5. Remove the first list element and record a checkpoint.
6. Delete the list and record a checkpoint.

The returned list contains four deltas, one for each checkpoint after the initial measurement.

The implementation must use `sys.getrefcount` correctly by accounting for the extra argument reference.

## Example

```python
result = predict_refcount_deltas()
# Example shape:
# [1, 2, 1, 0]
```

The exact values are produced by the CPython reference-counting behavior described above.

## What the gate checks

The gate runs a CPython oracle that performs the same binding sequence and measures
`sys.getrefcount` values with the argument artifact removed. The submitted function
must return the same delta vector as the oracle.

The gate uses exact equality because the environment is pinned to CPython 3.12.
