## Context

When exporting a PyTorch model to ONNX, each high‑level operation must be translated into an equivalent ONNX operator. Some PyTorch ops have a one‑to‑one mapping in the ONNX specification; others do not and must be decomposed into a sequence of primitive ONNX ops (e.g., `conv2d` is represented by a combination of `MatMul`, `Add`, and reshapes). Knowing which ops can be exported directly is essential for building efficient export pipelines.

## Task

Implement the function `map_ops(ops_list)` that receives a list of PyTorch operation names as strings and returns a list of booleans. For each input op, the corresponding boolean should be:

- `True` if the op has a direct ONNX operator in the current opset.
- `False` if the op must be decomposed into primitive ONNX ops.

The function signature is:

```python
def map_ops(ops_list: list[str]) -> list[bool]:
    ...
```

You may assume that all names in `ops_list` are valid PyTorch operation identifiers. The output list must have the same length and order as the input list.

## Example

```python
>>> ops = ["add", "conv2d", "relu", "batch_norm"]
>>> map_ops(ops)
[True, False, True, False]
```

In this example, `add` and `relu` are directly supported by ONNX, whereas `conv2d` and `batch_norm` require decomposition.

## What the gate checks

The grader compares each element of your output list with a reference oracle that maps operation names to their direct‑ONNX status. The comparison uses the metric **exact_match**: all booleans must match exactly for the submission to pass. No other metrics are evaluated.
