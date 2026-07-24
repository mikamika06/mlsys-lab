## Context

In a feed‑forward neural network with $L$ layers the forward pass produces an activation vector at each layer. If we store all activations during training, the peak memory is simply the sum of their sizes. Gradient checkpointing stores only a subset of these tensors and recomputes the rest on demand, reducing the peak memory.

Let $s_i$ be the number of neurons in layer $i$, and let $\mathrm{bytes}(x)=8\,x$ for 64‑bit floats. The full memory is
$$M_{\text{full}}=\sum_{i=0}^{L-1} \mathrm{bytes}(s_i).$$

If we checkpoint every $k$‑th layer, the set of stored tensors is $\{i: i\equiv 0\bmod k\}$ and the peak memory becomes
$$M_{\text{chkpt}}=\sum_{i=0}^{L-1}\mathbf{1}_{\,i\equiv 0 \bmod k}\,\mathrm{bytes}(s_i).$$

The ratio $R=M_{\text{full}}/M_{\text{chkpt}}$ quantifies the memory savings.

## Task

Implement `activation_memory_ratio(layer_sizes, checkpoint_every)` that returns this ratio as a Python float. The function receives:

- `layer_sizes`: list of integers, one per layer (including input and output).
- `checkpoint_every`: integer ≥ 1 indicating every how many layers to store an activation.

The implementation must be pure Python/NumPy, no external libraries beyond the standard library and NumPy.

## Example

```python
>>> from your_module import activation_memory_ratio
>>> activation_memory_ratio([10, 20, 30], 2)
1.5
```

Here $M_{\text{full}}=(10+20+30)\times8=480$ bytes and $M_{\text{chkpt}}=(10+30)\times8=320$ bytes, so the ratio is $480/320 = 1.5$.

## What the gate checks

The grader computes the analytic ratio for a set of test cases and compares your result to it with a relative error tolerance of $10^{-9}$. A correct implementation will pass all tests.
