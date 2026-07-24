## Context

In transformer architectures, attention mechanisms are parameterized by the number of query heads $n_q$ and key/value heads $n_{kv}$. The ratio
$$\frac{n_q}{n_{kv}}$$
determines both the classification of the scheme and a repeat factor used in downstream computations.

- If $n_{kv}=1$, the configuration is **MQA** (monolithic query attention).
- If $n_q=n_{kv}$, it is **MHA** (multi‑head attention).
- Otherwise it is **GQA** (global query attention).

The repeat factor is defined as
$$ n_{\text{rep}} = \frac{n_q}{\,n_{kv}\,}. $$

## Task

Implement `classify_and_compute_n_rep(config)`:

```python
def classify_and_compute_n_rep(config: dict) -> tuple[str,int]:
    ...
```

`config` will contain integer keys `'n_q'` and `'n_kv'`. The function must return a two‑tuple `(label, n_rep)` where `label` is one of the strings `"MHA"`, `"GQA"`, or `"MQA"` and `n_rep` is an integer equal to $n_q / n_{kv}$.

## Example

```python
config = {"n_q": 8, "n_kv": 2}
label, n_rep = classify_and_compute_n_rep(config)
# label == "GQA"
# n_rep == 4
```

## What the gate checks

The grader verifies that the returned tuple matches the exact reference for a set of test configurations. No other metrics are required.
