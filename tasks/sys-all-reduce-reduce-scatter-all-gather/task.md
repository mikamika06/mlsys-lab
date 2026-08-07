## Context

In distributed computing, collective operations coordinate data across $n$
processes.  The **all-reduce** takes a vector from every process and computes an
element-wise reduction (sum, max, …) that every process receives back:

$$y = x_0 \oplus x_1 \oplus \cdots \oplus x_{n-1}, \qquad
y_i = \bigoplus_{j=0}^{n-1} x_j[i]
\quad \text{for every element } i,$$

where $\oplus$ is applied position-wise (e.g.\ $+$ or $\max$).  Every process
receives the full result $y \in \mathbb{R}^{n \cdot k}$.

A standard identity decomposes this into two cheaper steps.

**Reduce-scatter.**  Split each vector into $n$ contiguous chunks of length $k$.
After reduction, process $j$ holds chunk $j$ of the result:

$$r_j = \bigoplus_{i=0}^{n-1} x_i^{(j)}, \qquad
x_i^{(j)} = x_i[jk : (j+1)k].$$

**All-gather.**  Each process shares its reduced chunk; every process collects
all chunks:

$$y = \bigl[\, r_0 \;\|\; r_1 \;\|\; \cdots \;\|\; r_{n-1} \,\bigr].$$

Composing the two gives $\text{all-gather}\!\bigl(\text{reduce-scatter}(x_0,
\ldots, x_{n-1})\bigr) = y$, which is exactly the all-reduce.

## Task

Implement two functions:

```python

def reduce_scatter(data, op='sum'):
    """
    Perform a reduce-scatter over n processes.

    Parameters
    ----------
    data : list of n arrays, each of shape (n * k,).
           data[i] is the vector contributed by process i.
    op   : "sum" or "max" — the element-wise reduction operator.

    Returns
    -------
    result : list of n arrays, each of shape (k,).
             result[j] = op over data[i][j*k:(j+1)*k] for all i in 0..n-1.
    """
    ...

def all_gather(data: list[list[float]]) -> list[list[float]]:
    """
    Perform an all-gather over n processes.

    Parameters
    ----------
    data : list of n arrays, each of shape (k,).
           data[j] is the chunk contributed by process j.

    Returns
    -------
    result : list of n arrays, each of shape (n * k,).
result[i] = data[0] + ... + data[n-1] for every i.
    """
    ...
```

Use only Python.  The composition must be correct: for any input,
`all_gather(reduce_scatter(data, op))` must equal the direct element-wise
reduction across all processes.

## Example

```python

x0 = [1, 2, 3, 4]   # n = 2, k = 2
x1 = [5, 6, 7, 8]

scattered = reduce_scatter([x0, x1], op="sum")
# chunk 0: [1+5, 2+6] = [6, 8]     → process 0
# chunk 1: [3+7, 4+8] = [10, 12]   → process 1

gathered = all_gather(scattered)
# both processes receive [6, 8, 10, 12]
# which equals the direct all-reduce [1+5, 2+6, 3+7, 4+8].
```

## What the gate checks

The gate computes a Python reference for the full all-reduce and then composes
the learner's `reduce_scatter` and `all_gather` on the same inputs.  It reports
the maximum absolute error across all test configurations:

$$\text{max\_abs\_err} = \max_{\text{cases}}\;
\max_{i \in \{0,\dots,n-1\}}\;
\bigl\lVert \text{all\_gather}\bigl(\text{reduce-scatter}(\mathbf{x})\bigr)_i
\;-\; y \bigr\rVert_{\infty}$$

The gate passes when $\text{max\_abs\_err} < 10^{-6}$.  Test cases cover both
`"sum"` and `"max"` reductions with $n \in \{2, 3, 4, 5, 6\}$ and
$k \in \{2, 3, 4, 5, 8\}$, seeded for determinism.
