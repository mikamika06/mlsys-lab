## Context

Pipeline parallel training splits a model into stages. In a 1F1B schedule, each stage alternates forward and backward work after the pipeline is filled. The pipeline bubble is the fraction of time where stages are idle because work has not reached them or because the schedule is draining.

For an interleaved schedule, each physical stage owns $v$ virtual stages. Increasing the number of virtual stages reduces the effective startup and drain cost because more useful work can be placed into the pipeline.

This task uses the idealized bubble model:

$$
\mathrm{bubble} = \frac{\mathrm{idle\ slots}}{\mathrm{active\ slots} + \mathrm{idle\ slots}} .
$$

For $p$ physical stages, $m$ microbatches, and $v$ virtual stages per physical stage, the model counts the startup/drain overhead as

$$
\mathrm{idle\ slots} = p - 1
$$

and the useful interleaved work as

$$
\mathrm{active\ slots} = m v .
$$

Therefore the expected fraction is

$$
\mathrm{bubble} = \frac{p-1}{m v + p - 1}.
$$

The function returns this fraction as a floating point value.

## Task

Implement `interleaved_1f1b_bubble_fraction(stages, microbatches, virtual_stages)`.

Arguments:

- `stages`: the number of physical pipeline stages $p$.
- `microbatches`: the number of microbatches $m$.
- `virtual_stages`: the number of virtual stages per physical stage $v$.

Return the idealized interleaved 1F1B bubble fraction as a `float`.

All inputs are positive integers. Do not simulate tensors or perform training operations.

## Example

```python
bubble = interleaved_1f1b_bubble_fraction(4, 8, 2)
print(bubble)
# 0.15789473684210525
```

## What the gate checks

The gate builds an oracle from the idle-slot and active-slot accounting model and compares the returned value using relative error.

The relative error

$$
\mathrm{rel\_err} =
\frac{|\hat{x}-x|}{|x|+10^{-12}}
$$

must be less than $10^{-9}$.
