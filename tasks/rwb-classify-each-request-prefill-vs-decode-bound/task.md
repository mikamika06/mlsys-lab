## Context

In transformer‑based language models the cost of a **prefill** step (processing an input prompt) grows quadratically with the prompt length $p$ because every token attends to all previous tokens:
$$C_{\text{prefill}}(p)=p^2.$$

During **decode**, each new token is generated one at a time.  The cost of generating $g$ tokens after a prompt of length $p$ grows linearly with the total sequence length that must be attended to for every generation step:
$$B_{\text{decode}}(p,g)=g\,(p+g).$$

A request is considered **prefill‑bound** when the prefill compute demand dominates the bandwidth‑bound decode cost, i.e.
$$C_{\text{prefill}}\ge B_{\text{decode}},$$
otherwise it is **decode‑bound**.

## Task

Implement a function that classifies each request in a batch according to the rule above.

```python
def classify_prefill_decode(requests):
    """
    Parameters
    ----------
    requests : list of dict
        Each dictionary must contain integer keys 'prompt_len' and 'gen_len'.

    Returns
    -------
    labels : list of str
        For each request, return the string "prefill" if
        prompt_len**2 >= gen_len * (prompt_len + gen_len),
        otherwise return "decode".
    """
```

The function must be pure Python; no external libraries are required.

## Example

```python

requests = [
    {"prompt_len": 10, "gen_len": 5},
    {"prompt_len": 20, "gen_len": 15},
    {"prompt_len": 3,  "gen_len": 30}
]

labels = classify_prefill_decode(requests)
print(labels)   # ['prefill', 'decode', 'decode']
```

## What the gate checks

The grader computes the expected labels with the same rule using Python for numerical stability.  
Your output must match exactly; otherwise the `exact_match` metric will be 0.0.
