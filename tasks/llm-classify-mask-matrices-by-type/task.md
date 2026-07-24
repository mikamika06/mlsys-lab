## Context

During self-attention, a boolean mask matrix dictates which tokens can attend to which other tokens. A `True` value at index `(i, j)` means token $i$ can attend to token $j$. Various model architectures employ different masking strategies:

1. **Bidirectional**: Every token attends to every other token. The mask is entirely `True`. (Used in BERT or ViT).
2. **Causal**: Tokens can only attend to themselves and past tokens. The mask is a lower-triangular matrix (`True` where $i \ge j$, `False` elsewhere). (Used in standard GPT).
3. **Sliding-Window (Causal)**: Tokens attend to themselves and up to $w$ previous tokens. The mask is a banded lower-triangular matrix (`True` where $0 \le i - j \le w$, for some $0 < w < N-1$). (Used in Mistral/Longformer).
4. **Prefix-LM**: A hybrid mask where a prefix of length $P$ ($0 < P < N$) is fully bidirectional (all tokens in the prefix can see each other), and all subsequent tokens follow standard causal masking. (`True` if $i \ge j$ OR if $i < P$ and $j < P$). (Used in T5 or UL2).

## Task

Write `classify_masks(masks)`:

```python
import numpy as np

def classify_masks(masks: list[np.ndarray]) -> list[str]:
    ...
```

Given a list of 2D boolean NumPy arrays of shape `(N, N)`, return a list of strings labeling each mask as one of `"bidirectional"`, `"causal"`, `"window"`, or `"prefix-lm"`. You may assume all input masks perfectly match one of these four categories for some valid parameter ($w$ or $P$).

## Example

```python
import numpy as np

# A 4x4 Causal Mask
causal_mask = np.array([
    [ True, False, False, False],
    [ True,  True, False, False],
    [ True,  True,  True, False],
    [ True,  True,  True,  True]
])

# A 4x4 Prefix-LM Mask with P=2
prefix_mask = np.array([
    [ True,  True, False, False],
    [ True,  True, False, False],
    [ True,  True,  True, False],
    [ True,  True,  True,  True]
])

classify_masks([causal_mask, prefix_mask])
# Returns: ["causal", "prefix-lm"]
```

## What the gate checks

The grader will generate multiple randomized test cases covering all 4 types of masks with various matrix sizes, window sizes, and prefix lengths. It tests if your returned list of string labels is an `exact_match` against the reference.
