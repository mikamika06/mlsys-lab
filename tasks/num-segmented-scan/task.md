## Context

A prefix scan transforms a sequence into cumulative values. For an inclusive sum scan over a vector
$x$, the output $y$ satisfies

$$
y_i = \sum_{j=0}^{i} x_j .
$$

A segmented scan adds reset points. Given segment-start flags $s$, where $s_i = 1$ means that
position $i$ begins a new segment, the sum restarts at every flagged position. The output is

$$
y_i =
\begin{cases}
x_i, & \text{if } s_i = 1, \\
y_{i-1} + x_i, & \text{otherwise}.
\end{cases}
$$

Segmented scans are useful in parallel algorithms because independent segments can be processed
without mixing state between groups.

## Task

Implement `segmented_scan(values, starts)`:

```python
def segmented_scan(values, starts):
    ...
```

The function receives two one-dimensional integer sequences of equal length.

- `values[i]` is the value to include in the scan.
- `starts[i]` is a flag. A nonzero value means that index `i` starts a new segment.

Return a list containing the inclusive segmented prefix sums. The first element of every segment
must equal its corresponding input value.

## Example

```python
values = [3, 1, 2, 5, 4, 1]
starts = [1, 0, 0, 1, 0, 0]

result = segmented_scan(values, starts)
# [3, 4, 6, 5, 9, 10]
```

The first segment is $[3, 1, 2]$ and scans to $[3, 4, 6]$. The second segment is
$[5, 4, 1]$ and scans independently to $[5, 9, 10]$.

## What the gate checks

The gate computes the expected result with an independent reference algorithm that applies the
segmented recurrence directly. The submitted function is tested on multiple integer inputs and
segment flag patterns.

The `exact_match` score must equal $1.0$. Any implementation that fails to reset the running
sum at segment boundaries will produce different values and will not pass.
