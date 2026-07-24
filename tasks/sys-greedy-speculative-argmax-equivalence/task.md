## Context

In natural language generation, a common strategy is greedy decoding: at each step choose the token with highest probability (or logit). Speculative decoding attempts to accelerate this by first generating a draft sequence using a smaller model and then verifying each token against a larger, more accurate model. If the draft token matches the large‑model argmax, it is accepted; otherwise we fall back to the large‑model choice.

Let $L^{(d)}_t \in \mathbb{R}^V$ denote the logits of the draft model at step $t$, and $L^{(t)}_t \in \mathbb{R}^V$ those of the target model. The greedy argmax for a single step is

$$
\operatorname{argmax}(L_t) = \underset{i}{\operatorname{arg\,max}}\, L_{t,i}.
$$

Speculative decoding produces a token sequence $s_1,\dots,s_T$ by the rule

$$
s_t =
\begin{cases}
\operatorname{argmax}\!\bigl(L^{(d)}_t\bigr) & \text{if }\operatorname{argmax}\!\bigl(L^{(d)}_t\bigr)=\operatorname{argmax}\!\bigl(L^{(t)}_t\bigr),\\[4pt]
\operatorname{argmax}\!\bigl(L^{(t)}_t\bigr) & \text{otherwise.}
\end{cases}
$$

The task is to implement this rule in a vectorised way.

## Task

Implement the function `greedy_speculative(draft_logits, target_logits)`:

```python
def greedy_speculative(draft_logits: np.ndarray,
                       target_logits: np.ndarray) -> List[int]:
    ...
```

`draft_logits` and `target_logits` are 2‑D NumPy arrays of shape `(T, V)` where `T` is the number of decoding steps and `V` the vocabulary size. The function must return a list of length `T` containing the token indices chosen by speculative decoding as defined above. Use only NumPy operations; no explicit Python loops over tokens.

## Example

```python
import numpy as np

draft = np.array([[0.1, 2.5, 0.3],
                  [1.0, 0.2, 0.9]])
target = np.array([[0.4, 2.6, 0.2],
                   [0.8, 0.1, 1.1]])

tokens = greedy_speculative(draft, target)
print(tokens)   # [1, 2]
```

Explanation:  
- Step 0: both models agree on token 1 (the largest logit).  
- Step 1: draft chooses token 0 but target’s argmax is token 2, so we fall back to 2.

## What the gate checks

The grader computes the reference greedy sequence from `target_logits` and compares it with the output of your implementation. The metric `spec_agreement` must equal `1.0`, meaning every token matches the true greedy choice. No other performance constraints are imposed, but a fully vectorised solution is expected.
