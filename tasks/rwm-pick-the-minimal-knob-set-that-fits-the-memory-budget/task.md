## Context

Large model training systems often have several memory-saving knobs. A runtime can choose to offload parameters, checkpoint intermediate activations, or offload activations. Each knob reduces the estimated peak memory usage but adds overhead such as communication or recomputation cost.

For a model with parameter memory $P$ and activation memory $A$, the baseline estimated peak memory is

$$M_0 = P + A.$$

Each memory-saving technique changes the peak estimate. The runtime should prefer the lowest-overhead configuration that satisfies a device memory budget $B$. This is a constrained selection problem:

$$\text{choose the first configuration } S \text{ such that } M(S) \le B.$$

Production systems typically check candidate configurations in a fixed preference order rather than searching for an arbitrary combination.

## Task

Implement `pick_knobs(param_bytes, activation_bytes, budget_bytes)`.

```python
def pick_knobs(param_bytes: int, activation_bytes: int, budget_bytes: int) -> tuple[str, ...]:
    ...
```

Return a tuple containing the selected memory-saving knobs. Valid knob names are:

- `"param-offload"`
- `"checkpoint"`
- `"activation-offload"`

The candidates must be evaluated in this order, from lowest overhead to highest overhead:

1. `()`
2. `("param-offload",)`
3. `("checkpoint",)`
4. `("activation-offload",)`
5. `("param-offload", "checkpoint")`
6. `("param-offload", "activation-offload")`
7. `("checkpoint", "activation-offload")`
8. `("param-offload", "checkpoint", "activation-offload")`

Use the following peak memory model:

- `"param-offload"` removes $80\%$ of parameter memory.
- `"checkpoint"` removes $60\%$ of activation memory.
- `"activation-offload"` removes $90\%$ of activation memory.

When multiple knobs are present, apply every reduction. For example, the memory for `("param-offload", "checkpoint")` is

$$0.2P + 0.4A.$$

Return the first candidate whose estimated peak memory is less than or equal to `budget_bytes`. If even the final candidate does not fit, return the full three-knob tuple.

## Example

```python
knobs = pick_knobs(
    param_bytes=800,
    activation_bytes=200,
    budget_bytes=500,
)

# ("param-offload",)
```

The baseline memory is $1000$. Parameter offload reduces it to

$$0.2(800) + 200 = 360,$$

which fits the budget, so later higher-overhead options are not considered.

## What the gate checks

The gate computes the oracle choice by evaluating every candidate subset using the same memory estimator and the required increasing-overhead order. It tests many random model sizes and memory budgets.

The returned tuple must exactly match the oracle-selected knob set. Returning a configuration that also fits is insufficient if it is not the minimal-overhead choice.
