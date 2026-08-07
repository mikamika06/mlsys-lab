## Context

In scaled‑dot‑product attention (SDPA) the logits $QK^\top / \sqrt{d_k}$ are modified by a mask before the softmax. The mask can be *causal*—disallowing future positions—or an explicit tensor that either blocks positions (boolean mask) or adds bias values (float mask). In many libraries these options are mutually exclusive: a causal flag cannot coexist with an explicit mask.

The classification of which masking path is taken for a given pair of arguments $(\texttt{is\_causal}, \texttt{attn\_mask})$ is useful for debugging and for generating reproducible attention patterns. The task below asks you to implement a small dispatcher that, given the two inputs, returns one of four labels:

- `"causal"` – causal masking only,
- `"bool_mask"` – an explicit boolean mask,
- `"float_mask"` – an explicit numeric mask (added as bias),
- `"none"` – no masking at all,
- `"illegal"` – a combination that is not allowed by the library’s API.

The mapping follows the rules used in the reference implementation of the production attention routine:

1. If `is_causal` is `True` and `attn_mask` is `None`, return `"causal"`.
2. If `is_causal` is `False` and `attn_mask` is `None`, return `"none"`.
3. If `attn_mask` is a list of boolean type, return `"bool_mask"`.
4. If `attn_mask` is a list of numeric type (int or float), return `"float_mask"`.
5. Any other combination (e.g., `is_causal=True` together with an explicit mask) returns `"illegal"`.

The function must be pure: no side effects, and it should raise no exceptions for the inputs described above.

## Task

Implement the following function:

```python
def classify_masking(is_causal: bool, attn_mask: list | None) -> str:
    """
    Return a string describing which masking path SDPA will take.
    """
```

The function must follow exactly the mapping described in the context section.

## Example

```python

print(classify_masking(True, None))
# 'causal'

print(classify_masking(False, None))
# 'none'

print(classify_masking(False,
                       [[True, False], [False, True]]))
# 'bool_mask'

print(classify_masking(False,
                       [[0.0, -1e9], [-1e9, 0.0]]))
# 'float_mask'

print(classify_masking(True,
                       [[True, False], [False, True]]))
# 'illegal'
```

## What the gate checks

The grader computes the expected classification using a reference implementation that follows the same rules as above and compares it to your output. The comparison is case‑sensitive; any deviation causes the gate to fail.

No numeric tolerance or performance metrics are involved—only exact string equality.
