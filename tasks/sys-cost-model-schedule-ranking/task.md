## Context

Compiler autotuners such as XLA and TVM search over many possible schedules for the
same computation. A schedule describes choices such as tile sizes or loop
parameters, and a cost model predicts which candidate should run fastest.

For tiled matrix multiplication with output size $M \times N$ and reduction size
$K$, a simple proxy can estimate the amount of work from the number of tiles:

$$
\mathrm{tiles} =
\left\lceil \frac{M}{t_m} \right\rceil
\left\lceil \frac{N}{t_n} \right\rceil
\left\lceil \frac{K}{t_k} \right\rceil .
$$

A schedule ranking system sorts candidates by increasing predicted cost. Equal
costs are resolved by the candidate identifier to keep the result deterministic.

## Task

Implement `rank_schedules(candidates, shape)`:

```python
def rank_schedules(candidates, shape):
    ...
```

`shape` is a tuple `(M, N, K)`. `candidates` is a list of dictionaries. Each
dictionary contains:

- `id`: a unique string identifier.
- `tile_m`, `tile_n`, `tile_k`: positive integer tile sizes.

Return a list of candidate identifiers ordered from lowest cost to highest cost.

The cost model must use the tiled matrix multiplication proxy:

$$
\mathrm{cost} =
\left\lceil \frac{M}{t_m} \right\rceil
\left\lceil \frac{N}{t_n} \right\rceil
\left\lceil \frac{K}{t_k} \right\rceil
t_m t_n t_k .
$$

Use integer arithmetic and return only the ordered identifiers.

## Example

```python
candidates = [
    {"id": "a", "tile_m": 8, "tile_n": 8, "tile_k": 8},
    {"id": "b", "tile_m": 16, "tile_n": 16, "tile_k": 16},
]

rank_schedules(candidates, (64, 64, 64))
# ["b", "a"]
```

## What the gate checks

The gate computes the reference ordering independently from the same cost-model
definition and compares the returned identifier list exactly. A candidate fails
if it uses a different ranking rule, ignores tile parameters, or returns a
different tie ordering.
