## Context

CPython uses different allocators depending on the requested size. The `pymalloc` allocator manages small object memory using arenas, pools, and fixed-size blocks. Larger requests are handled by the platform allocator.

The transition boundary is an implementation detail of the running interpreter. A classifier should inspect the active CPython build instead of assuming that every interpreter uses the same allocator policy.

For a requested size $s$ bytes, the classifier returns whether the request belongs to a size class managed by `pymalloc`.

## Task

Implement `classify_pymalloc(sizes)`:

```python
def classify_pymalloc(sizes: list[int]) -> list[bool]:
    ...
```

The function receives a list of integer byte sizes and returns a list of booleans with the same length. Each result must be `True` when the current CPython interpreter uses `pymalloc` for that request size and `False` when it uses the raw allocator path.

Determine the behavior from runtime CPython information. Do not assume a fixed boundary.

## Example

```python
sizes = [496, 512, 513]

result = classify_pymalloc(sizes)

# Example on a CPython build with the usual small-object allocator:
# [True, True, False]
```

The exact result depends on the interpreter build.

## What the gate checks

The gate computes the expected result from the running CPython interpreter using allocator metadata exposed by CPython itself. It then compares the submitted function output exactly.

The `exact_match` metric must be $1.0$. A solution that only returns a guessed constant threshold instead of reading runtime allocator information will fail on boundary cases.
