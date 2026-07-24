## Context

Transformer blocks are typically built around a residual connection of the form  

$$
y = x + f(x),
$$

where $f$ is some nonlinear transformation (e.g., multi‑head attention followed by a feed‑forward network).  
Layer normalisation can be inserted either **before** the nonlinearity, **after**, or in both places:

* **Pre‑norm** – normalise the input to $f$:  

  $$y = x + f(\operatorname{Norm}(x)).$$

* **Post‑norm** – normalise the output of the residual addition:  

  $$y = \operatorname{Norm}\bigl(x + f(x)\bigr).$$

* **Sandwich** – combine both strategies:  

  $$y = \operatorname{Norm}\Bigl(x + f\bigl(\operatorname{Norm}(x)\bigr)\Bigr).$$

These three wiring patterns are widely used in practice and have different optimisation properties.  
Your task is to write a small classifier that, given the wiring description of several blocks, returns the appropriate label for each block.

## Task

Implement `classify_norm_wiring(blocks)`:

```python
def classify_norm_wiring(blocks: list[dict]) -> list[str]:
    ...
```

Each element in `blocks` is a dictionary with two boolean keys:

* `"pre_norm"` – whether the block uses pre‑norm.
* `"post_norm"` – whether the block uses post‑norm.

The function must return a list of strings, one per block, where each string is either `"pre"`, `"post"`, or `"sandwich"` according to the rules above.  
If a block has neither flag set, raise `ValueError`.

## Example

```python
blocks = [
    {"pre_norm": True,  "post_norm": False},
    {"pre_norm": False, "post_norm": True},
    {"pre_norm": True,  "post_norm": True}
]
labels = classify_norm_wiring(blocks)
print(labels)   # ['pre', 'post', 'sandwich']
```

## What the gate checks

The grader generates random block configurations and compares your output against a reference implementation that applies the same logic.  
It reports an `exact_match` score of 1.0 only if all labels match exactly; otherwise it returns 0.0.
