## Context

In many transformer architectures, Rotary Position Embeddings (RoPE) scale the base frequency table $\{\omega_i\}_{i=0}^{n-1}$ by a scalar factor $s>0$ to control the effective positional range.  
The modified frequencies are therefore

$$\omega'_i = s\,\omega_i,\qquad i=0,\dots,n-1.$$

In practice, due to numerical rounding or additional processing, each entry may deviate slightly from this exact scaling.

## Task

Implement `recover_scale_factor`:

```python
def recover_scale_factor(orig_freqs: list[float],
                         mod_freqs: list[float]) -> float:
    ...
```

The function receives two 1‑D list of equal length containing the original and modified frequencies, respectively. It must return a single Python `float` that estimates the true scaling factor $s$.

## Example

```python
orig = [0.5, 1.0, 2.0]
mod  = orig * 1.75
s_est = recover_scale_factor(orig, mod)
print(s_est)   # 1.75
```

## What the gate checks

The grader generates random test cases where `mod_freqs` are obtained from `orig_freqs` by multiplying each element with a common factor $s$ and adding a small perturbation to one entry.  
It then computes the relative error

$$\mathrm{rel\_err} = \frac{|\,\hat{s}-s\,|}{|s|}.$$

Your solution must achieve $\mathrm{rel\_err}\le 10^{-4}$ on all provided tests.
