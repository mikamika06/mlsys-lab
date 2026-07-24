## Context

Production ML runtimes often represent dynamic tensor dimensions with a range guard. A dimension specification such as `Dim(min, max)` allows a runtime dimension $n$ only when

$$
\mathrm{min} \leq n \leq \mathrm{max}.
$$

If a model is compiled with a range that is too narrow, a valid input can fail before execution because the runtime guard rejects the shape. The fix is to expand only the failing side of the range enough to include the observed input.

For a one-dimensional dynamic batch dimension, the corrected bounds are:

$$
\mathrm{new\_min} =
\begin{cases}
n & n < \mathrm{min} \\
\mathrm{min} & \text{otherwise}
\end{cases}
$$

and

$$
\mathrm{new\_max} =
\begin{cases}
n & n > \mathrm{max} \\
\mathrm{max} & \text{otherwise}.
\end{cases}
$$

After the guard is repaired, the runtime can execute the operation again.

## Task

Implement `fix_shape_range_and_run(lower, upper, x)`.

The arguments are:

- `lower`: the current minimum allowed size for the first dimension.
- `upper`: the current maximum allowed size for the first dimension.
- `x`: a NumPy array whose first dimension is the runtime shape that must be checked.

The function must return:

```python
((new_lower, new_upper), output)
```

where:

- `(new_lower, new_upper)` is the smallest corrected range that accepts `x.shape[0]`.
- `output` is the result of re-running the operation after the guard is fixed.

The runtime operation is defined as converting `x` to `float64` and applying:

$$
y = 2x + 1.
$$

Use NumPy for the numerical operation. The returned output must be a NumPy array with `float64` dtype.

## Example

```python
import numpy as np

x = np.ones((7, 2), dtype=np.float32)

fixed, y = fix_shape_range_and_run(4, 6, x)

# fixed == (4, 7)
# y is:
# [[3. 3.]
#  [3. 3.]
#  ...
#  [3. 3.]]
```

## What the gate checks

The gate creates runtime failures where the observed dimension violates either the lower or upper guard. It computes the oracle repair by applying the range correction algorithm and then runs the NumPy reference operation.

The returned range must exactly match the oracle range. The returned output is compared with the oracle output using maximum absolute error after the repaired execution. A solution that widens unnecessary bounds or changes the numerical operation fails.
