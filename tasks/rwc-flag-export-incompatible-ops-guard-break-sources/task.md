## Context

When exporting a PyTorch model with `torch.export`, the compiler must be able to reason about the graph statically. Operations that depend on runtime data or cause side effects break this assumption. These are called *guard* or *break* sources. Typical examples include:

- Data‑dependent calls such as `$x.item()$, $x.nonzero(),$ and `$bool(x)$.
- In‑place mutations like `$x.add_()$` or any operation whose name ends with an underscore.
- Python side‑effects, e.g. a call to `print`, a lambda that captures mutable state, or any function containing the substring `"python"`.

A clean export requires all operations in the graph to be *export‑safe*. The task is to identify which operations are incompatible.

## Task

Implement the following function:

```python
def flag_export_incompatible_ops(ops: list[str]) -> list[bool]:
    """
    Given a list of operation names, return a list of booleans indicating whether each
    operation is export‑incompatible (True) or export‑safe (False).

    The rules are:
      * If the name contains "item", "nonzero", or "bool" → incompatible.
      * If the name ends with "_" (in‑place mutation) → incompatible.
      * If the name contains the substring "python" → incompatible.
      * Otherwise → compatible.

    The function must preserve order and return a list of booleans of the same length
    as `ops`.  It should not modify the input list.
    """
```

## Example

```python
>>> ops = ["add", "sub_", "item", "nonzero", "python_side_effect", "mul"]
>>> flag_export_incompatible_ops(ops)
[False, True, True, True, True, False]
```

## What the gate checks

The grader computes the expected labels using the same rule set and compares them to
the student's output.  The metric `exact_match` must be `1.0`.  No other metrics are used.
