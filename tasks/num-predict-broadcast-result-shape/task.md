## Context

Broadcasting is a set of rules that NumPy uses to perform arithmetic operations on arrays with different shapes. The key idea is that dimensions of size $1$ can be stretched to match the corresponding dimension of another array, and missing leading dimensions are treated as $1$. Formally, two shapes $\mathbf{a}=(a_1,\dots,a_m)$ and $\mathbf{b}=(b_1,\dots,b_n)$ are broadcastable if for each trailing index $k$ (counting from the end) either $a_k=b_k$, or one of them equals $1$. The resulting shape is obtained by taking, for each such pair, the maximum of the two sizes.

## Task

Implement `broadcast_shape(shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> Tuple[int, ...]` that returns the broadcast result shape according to NumPy's rules. If the shapes are not compatible, return an empty tuple `()`.

The function should accept any two tuples of positive integers (or zero). It must be pure Python/NumPy; no external libraries.

## Example

```python
>>> broadcast_shape((3,1), (2,4))
(3, 2, 4)
>>> broadcast_shape((5,), (1,5,1))
(1, 5, 5)
>>> broadcast_shape((2,3), (4,))
()
```

## What the gate checks

The grader computes the expected shape using `numpy.broadcast_shapes` and compares it to your output. The metric `exact_match` must be 1.0 for all test cases.
