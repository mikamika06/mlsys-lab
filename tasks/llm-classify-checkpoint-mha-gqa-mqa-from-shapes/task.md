## Context

Transformer models use varying attention grouping schemes to optimize inference speed and memory. In all schemes, the query ($Q$), key ($K$), and value ($V$) projections share the same per-head dimension $d_{\text{head}}$.

Let $n_q$ be the number of query heads, and $n_{kv}$ be the number of key/value heads. The three main schemes are:
1. **MHA (Multi-Head Attention)**: Each query head has its own key/value head. ($n_{kv} = n_q$)
2. **MQA (Multi-Query Attention)**: All query heads share a single key/value head. ($n_{kv} = 1$)
3. **GQA (Grouped-Query Attention)**: Query heads are divided into groups, and each group shares one key/value head. ($1 < n_{kv} < n_q$, and $n_q$ is divisible by $n_{kv}$)

A weight matrix for a linear projection from $d_{\text{in}}$ to an attention component with $n$ heads has an output dimension of $d_{\text{out}} = n \times d_{\text{head}}$.

## Task

Write `classify_attention(wq_shape, wk_shape, wv_shape, n_q)`:

```python
def classify_attention(
    wq_shape: tuple[int, int],
    wk_shape: tuple[int, int],
    wv_shape: tuple[int, int],
    n_q: int
) -> tuple[str, int]:
    ...
```

Given the shapes `(in_features, out_features)` for the $Q, K, V$ projection matrices and the number of query heads $n_q$, determine the attention scheme and the number of key/value heads. 

Return a tuple: `(scheme, n_kv)`, where `scheme` is one of `"MHA"`, `"MQA"`, or `"GQA"`.
You may assume the shapes are valid and $d_{\text{head}}$ is consistent across $Q, K, V$. The key and value heads will always have the same output dimension.

## Example

```python
# LLaMA 1 65B style (MHA)
# d_model = 8192, n_q = 64, d_head = 128
classify_attention((8192, 8192), (8192, 8192), (8192, 8192), 64)
# Returns: ("MHA", 64)

# LLaMA 2 70B style (GQA)
# d_model = 8192, n_q = 64, d_head = 128, n_kv = 8
classify_attention((8192, 8192), (8192, 1024), (8192, 1024), 64)
# Returns: ("GQA", 8)
```

## What the gate checks

The grader calls your function on various valid weight shapes corresponding to known open-source model configurations (e.g. LLaMA, Falcon, Mistral) and checks that both the string label and the extracted $n_{kv}$ exactly match the reference.
