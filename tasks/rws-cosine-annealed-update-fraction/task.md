## Context

The cosine-annealed update fraction is a schedule used in some optimization algorithms to adjust the learning rate or update fraction over time. Given an initial value $f_0$ and a total number of steps $T$, the update fraction at step $t$ is calculated as:

$$f_t = \frac{f_0}{2} \cdot (1 + \cos(\pi \frac{t}{T}))$$

This schedule starts with $f_0$, decreases to $0$ at the middle of the training, and then increases back to $f_0$ at the end.

## Task

Implement `cosine_annealed_update_fraction(f0, T, t, nnz)`:

```python
def cosine_annealed_update_fraction(f0, T, t, nnz):
    ...
```

It takes the initial update fraction `f0`, the total number of steps `T`, the current step `t`, and the number of non-zero elements `nnz`. The function should return the number of connections updated at step `t`.

## Example

```python
f0 = 0.1
T = 100
t = 50
nnz = 1000
updated_connections = cosine_annealed_update_fraction(f0, T, t, nnz)
```

## What the gate checks

The gate checks the `exact_match` metric, which compares the learner's output with the reference output. The reference output is calculated using a NumPy implementation of the cosine-annealed update fraction schedule. The gate passes if the learner's output matches the reference output.
