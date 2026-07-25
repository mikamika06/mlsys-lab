## Context

The roofline model describes the attainable performance of a kernel as the minimum of its compute‑bound and memory‑bound limits. For a given hardware platform we know two constants:
- $\mathrm{Peak}_{\text{compute}}$ – the maximum floating‑point operations per second,
- $\mathrm{Peak}_{\text{mem}}$ – the maximum bytes that can be transferred per second.

The ratio
$$
\theta = \frac{\mathrm{Peak}_{\text{compute}}}{\mathrm{Peak}_{\text{mem}}}
$$
is called the *roofline threshold*. A kernel with operational intensity (OI) larger than $\theta$ will run at its compute peak; otherwise it is limited by memory bandwidth.

In large‑language‑model decoding we can model each token as requiring a fixed number of floating‑point operations $C$ and a fixed amount of data movement $M$. When the same batch of $B$ sequences is processed together, the total work scales linearly with $B$, but the total data traffic grows only like $\sqrt{B}$ because many intermediate tensors can be reused across the batch. Consequently the overall operational intensity for decoding is
$$
\mathrm{OI}(B) = \frac{B\,C}{M\,\sqrt{B}} = \frac{C}{M}\,\sqrt{B}.
$$

The *crossover* batch size $B^{*}$ is the smallest integer such that $\mathrm{OI}(B^{*}) \ge \theta$.

## Task

Implement `crossover_batch_size`:

```python
def crossover_batch_size(peak_compute: float,
                         peak_mem: float,
                         compute_per_token: float,
                         mem_per_token: float) -> int:
    ...
```

The function should return the minimal integer batch size $B^{*}$ that satisfies
$$\frac{C}{M}\,\sqrt{B} \;\ge\; \theta,$$
where $\theta = \mathrm{Peak}_{\text{compute}}/\mathrm{Peak}_{\text{mem}}$.
Use only the Python standard library (you may import `math`).

## Example

```python
>>> crossover_batch_size(peak_compute=200e9,
...                       peak_mem=25e9,
...                       compute_per_token=1.0,
...                       mem_per_token=4.0)
2
```

Here $\theta = 8$, $C/M = 0.25$ and the smallest integer $B$ with $0.25\sqrt{B}\ge 8$ is $B=2$.

## What the gate checks

The grader evaluates a handful of random test cases. For each case it computes the reference answer using the exact formula above, then compares your output to that reference. The relative error
$$
\texttt{rel\_err} = \frac{|\,\text{your}\;B^{*} - B_{\text{ref}}^{*}|}{\max(1,B_{\text{ref}}^{*})}
$$
must be at most $10^{-3}$.

The function must run in constant time and use only the standard library. No external packages are required.
