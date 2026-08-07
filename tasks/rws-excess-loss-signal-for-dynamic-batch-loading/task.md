## Context

In dynamic batch loading systems, each domain or data source may exhibit a different training loss trajectory. To decide whether to allocate more samples from a particular domain, we often look at the *excess loss*:
$$e_i \;=\;\ell^{\text{curr}}_i \;-\;\ell^{\text{ref}}_i,$$
where $\ell^{\text{curr}}_i$ is the current loss for domain $i$ and $\ell^{\text{ref}}_i$ is a reference or baseline loss. A positive excess indicates that the domain is under‑performing relative to its baseline.

The task below asks you to implement a small utility that, given two 1‑D list of current and reference losses, returns the vector of excess losses.

## Task

Implement `excess_loss_signal`:

```python
def excess_loss_signal(current_losses: list[float], reference_losses: list[float]) -> list[float]:
    ...
```

The function should:

* Accept two 1‑D list of equal length.
* Return a new list containing the element‑wise difference `current - reference`.
* Preserve the input dtype as `float64`.

No loops, no external libraries beyond Python.

## Example

```python
curr = [0.25, 0.40, 0.10]
ref = [0.20, 0.35, 0.15]
excess = excess_loss_signal(curr, ref)
print(excess)  # [0.04999999999999999, 0.050000000000000044, -0.04999999999999999]
```

## What the gate checks

The grader computes a Python reference `curr - ref` and measures the global relative L2 error
$$\mathrm{rel\_err} \;=\;
\frac{\|\,\text{candidate} - \text{reference}\,\|}{\|\text{reference}\| + 10^{-12}}.$$
The solution must achieve $\mathrm{rel\_err} \le 1\times10^{-6}$ on a set of random test cases.
