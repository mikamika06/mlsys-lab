## Context

Python attribute lookup follows object lookup rules and may involve descriptors. A long attribute chain inside a repeated loop performs the same lookup sequence many times.

For example, the expression

```python
model.state.block.weight.value
```

requires multiple attribute lookups. If a loop repeats this expression for $T$ iterations, the number of lookup operations grows with $T$.

A common optimization is to bind stable values to local variables before the loop. The computation stays the same, but the interpreter performs fewer repeated attribute accesses.

## Task

Implement `accumulate_metric(model, steps)`.

The input object contains the nested attribute chain:

```python
model.state.block.weight.value
```

The function must return the sum of this value repeated `steps` times:

$$
\sum_{i=1}^{T} v = T \cdot v,
$$

where $v = \texttt{model.state.block.weight.value}$ and $T = \texttt{steps}$.

Optimize the implementation by avoiding repeated evaluation of the full attribute chain inside the loop. The result must exactly match the direct implementation.

## Example

```python
class Box:
    pass

model = Box()
model.state = Box()
model.state.block = Box()
model.state.block.weight = Box()
model.state.block.weight.value = 5

result = accumulate_metric(model, 3)
# result == 15
```

## What the gate checks

The gate first computes the expected value using a direct CPython attribute-access implementation and checks that the submitted function returns the same result.

It also traces the submitted function and counts actual CPython `LOAD_ATTR` opcode executions. The optimized solution must keep attribute lookup events under the fixed budget. A version that performs the deep attribute chain on every iteration will exceed the budget.
