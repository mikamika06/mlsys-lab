## Context

Production systems often group variable-sized objects into a small number of allocation buckets. A bucket of capacity $b$ can store every item with size $s \le b$. Items assigned to a larger bucket create padding waste:

$$
\mathrm{waste} = \sum_s h_s \left(\mathrm{bucket}(s)-s\right),
$$

where $h_s$ is the number of items with size $s$, and $\mathrm{bucket}(s)$ is the smallest selected bucket that can contain the item.

Each selected bucket consumes memory. If a bucket has capacity $b$ and each unit of capacity costs `unit` bytes, the memory cost is

$$
\mathrm{memory} = \sum_{b \in B} b \cdot \mathrm{unit}.
$$

The goal is to select a limited number of bucket sizes while staying under a memory cap and minimizing padding waste. This is a constrained selection problem similar to knapsack: choices have costs, and the final objective is evaluated from the selected set.

## Task

Implement:

```python
def choose_buckets(size_histogram, unit, cap, max_buckets):
    ...
```

`size_histogram` is a dictionary mapping positive integer object sizes to integer counts. `unit` is the memory cost of one bucket capacity unit. `cap` is the maximum total bucket memory. `max_buckets` is the maximum number of buckets that may be selected.

Return a tuple:

```python
(bucket_sizes, waste)
```

where `bucket_sizes` is a sorted list of selected bucket capacities and `waste` is the minimized padding waste.

The candidate bucket sizes are exactly the keys of `size_histogram`. Every input size must be covered by at least one selected bucket. If several solutions have the same minimum waste, return the lexicographically smallest sorted `bucket_sizes` list.

The selected buckets must satisfy:

$$
\sum_{b \in B} b \cdot \mathrm{unit} \le \mathrm{cap}
$$

and

$$
|B| \le \mathrm{max\_buckets}.
$$

## Example

```python
sizes = {3: 4, 5: 2, 8: 1}
buckets, waste = choose_buckets(sizes, 4, 40, 2)

# buckets == [5, 8]
# waste == 8
```

The bucket memory is $5 \cdot 4 + 8 \cdot 4 = 52$, so the example cap would actually reject that choice. A valid implementation must respect the memory cap and may choose a different feasible set.

## What the gate checks

The gate runs several hidden cases. It computes the optimal bucket set using a dynamic programming oracle over candidate bucket sizes, memory cost, and bucket count. The returned bucket list and the achieved waste must exactly match the oracle result.

A solution that uses a greedy rule such as always picking the largest sizes first can fail because a combination of smaller buckets may produce lower total padding while using the same memory budget.
