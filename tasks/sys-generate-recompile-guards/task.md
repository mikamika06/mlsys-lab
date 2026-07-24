## Context

In machine learning frameworks, especially those that utilize just-in-time compilation like PyTorch, it is crucial to determine when a recompile is necessary. A recompile is triggered when there is a change in the input shape, data type, or rank. This task involves generating recompile guards based on a given trace of inputs and a stream of new inputs.

The recompile condition can be mathematically expressed as follows: for an input tensor $x$, a recompile is triggered if any of the following conditions hold:

1. Shape change: $\text{shape}(x) \neq \text{shape}(x_{\text{new}})$
2. Data type change: $\text{dtype}(x) \neq \text{dtype}(x_{\text{new}})$
3. Rank change: $\text{rank}(x) \neq \text{rank}(x_{\text{new}})$

## Task

Implement `generate_recompile_guards(trace: List[Tuple[np.ndarray, str]], new_inputs: List[np.ndarray]) -> List[bool]`:

```python
def generate_recompile_guards(trace: List[Tuple[np.ndarray, str]], new_inputs: List[np.ndarray]) -> List[bool]:
    ...
```

The function takes a list of tuples `trace`, where each tuple contains an input tensor and its associated identifier (a string). It also takes a list of new input tensors. The function should return a list of booleans indicating whether a recompile is triggered for each new input based on the last input in the trace.

## Example

```python
import numpy as np

trace = [(np.array([[1, 2], [3, 4]]), "input1")]
new_inputs = [np.array([[1, 2], [3, 4]]), np.array([[1, 2, 3]]), np.array([[1.0, 2.0], [3.0, 4.0]])]
guards = generate_recompile_guards(trace, new_inputs)
# [False, True, True]
```

## What the gate checks

The gate checks whether the output of your function matches the expected output for a set of test cases. The exact match metric ensures that your implementation correctly identifies when a recompile is necessary based on the input conditions outlined above. The function must be able to handle various shapes, data types, and ranks of tensors.
