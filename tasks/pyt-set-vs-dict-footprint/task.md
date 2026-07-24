## Context

In CPython, both `set` and `dict` are implemented as hash tables that store key objects in an array of buckets.  
The memory footprint of each container depends on the number of entries it holds and on a fixed overhead per bucket.  
Because a set stores only keys while a dict stores key–value pairs, a dictionary typically consumes more memory for the same logical data.

The size of a Python object can be obtained with `sys.getsizeof`, which reports the number of bytes allocated for that object in CPython's heap.

## Task

Implement the function

```python
def set_dict_size_ratio(elements: Iterable[int]) -> float:
    ...
```

It receives an iterable of hashable elements (integers are sufficient).  
The function must create a `set` and a `dict` containing those elements, then return the ratio

$$\frac{\operatorname{getsizeof}(\text{set})}{\operatorname{getsizeof}(\text{dict})}\;.$$

The result should be a floating‑point number (`float`).  The function must not modify the input iterable.

## Example

```python
>>> from your_module import set_dict_size_ratio
>>> elements = [1, 2, 3]
>>> ratio = set_dict_size_ratio(elements)
>>> print(ratio)
0.6666666666666666   # example value; actual number depends on CPython version
```

## What the gate checks

The grader constructs a reference ratio using `sys.getsizeof` on a freshly built set and dict.  
Your implementation is accepted if its output matches the reference within a relative error of $10^{-9}$ for all test cases.
