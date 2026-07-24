## Context

In dynamic tracing frameworks such as PyTorch FX, a *graph* is built by recording the operations performed on tensors during a forward pass.  
A **graph break** occurs when the tracer encounters code that cannot be represented in the graph, for example data‑dependent control flow or unsupported Python constructs.  Detecting these breaks early allows developers to refactor code into traceable patterns.

## Task

Implement `classify_breaks`:

```python
def classify_breaks(snippets: list[str]) -> list[str]:
    ...
```

The function receives a list of Python code snippets (each snippet is a string containing either a lambda expression or a simple function definition).  
It must return a list of the same length where each element is either `"traceable"` if the snippet can be represented by a static graph, or `"break"` otherwise.  The classification should be based on syntactic analysis only; no execution of the snippets is allowed.

## Example

```python
snippets = [
    "lambda x: x + 1",
    "lambda x: x * y",          # uses undefined variable `y`
    "def f(x): return x + 2",
    "def g(x):\n    if x > 0:\n        return x\n    else:\n        return -x"
]
labels = classify_breaks(snippets)
# ['traceable', 'break', 'traceable', 'break']
```

## What the gate checks

The grader computes a ground‑truth classification for each snippet using an AST‑based oracle.  
Your output must match this reference exactly (`exact_match`).  No other metrics are evaluated.
