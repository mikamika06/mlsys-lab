## Context

Dynamic quantization in PyTorch is a lightweight post‑training technique that replaces floating‑point weights with integer representations for selected modules.  
Only a subset of module types support this conversion; the default implementation supports `torch.nn.Linear` and `torch.nn.LSTM`.  All other modules are skipped.

Let $t_i$ denote the type name of the *i*‑th module in a model.  We define an indicator mask
$$m_i = \begin{cases}
1 & \text{if } t_i\in\{\text{Linear},\text{LSTM}\}\\[4pt]
0 & \text{otherwise}
\end{cases}.$$
The task is to implement a function that, given a list of type names, returns this mask.

## Task

Implement `classify_quantizable`:

```python
def classify_quantizable(names: list[str]) -> list[int]:
    ...
```

It receives a list of module type names (strings) and must return a list of the same length containing 1 for each name that would be quantized by `torch.quantization.quantize_dynamic` under its default settings, and 0 otherwise.  The function should be case‑sensitive: only the exact strings `"Linear"` and `"LSTM"` are considered quantizable.

## Example

```python
>>> classify_quantizable(["Linear", "Conv2d", "LSTM"])
[1, 0, 1]
>>> classify_quantizable(["Embedding", "BatchNorm2d"])
[0, 0]
```

## What the gate checks

The grader computes a reference mask using the same rule above and compares it to the candidate’s output with an exact match metric.  The solution must return the correct integer values for all test cases; any mismatch or exception causes the gate to fail.
