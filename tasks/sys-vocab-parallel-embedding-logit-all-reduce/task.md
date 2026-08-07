## Context

Tensor parallel training splits large model tensors across ranks. A vocabulary-parallel
embedding table partitions rows of the vocabulary across devices. Rank $r$ owns a
range of vocabulary ids and only stores the corresponding embedding vectors.

For token ids $x$ and embedding matrix $E \in \mathbb{R}^{V \times H}$, the full
embedding lookup is

$$
Y_i = E_{x_i}.
$$

With vocabulary sharding, each rank creates a local result containing values only
for tokens in its vocabulary range. Summing these local results is equivalent to
an all-reduce operation:

$$
Y = \sum_{r=0}^{R-1} Y^{(r)}.
$$

The same idea can be used for logits. If rank $r$ owns output rows
$E_r \in \mathbb{R}^{V_r \times H}$, it produces logits only for its vocabulary
slice. An all-reduce sum of zero-filled partial tensors reconstructs the complete
logit matrix:

$$
L = \sum_{r=0}^{R-1} L^{(r)}.
$$

This task simulates the communication pattern using list on one machine.

## Task

Implement `vocab_parallel_forward(token_ids, embedding, output_weight, world_size)`.

Arguments:

- `token_ids`: a 1-D Python integer array with token ids in the range
  $0 \leq x_i < V$.
- `embedding`: a list of shape $(V, H)$ containing the full embedding
  table.
- `output_weight`: a list of shape $(V, H)$ containing output projection
  rows.
- `world_size`: the number of vocabulary shards.

Return a tuple `(hidden, logits)`:

- `hidden` is the gathered embedding lookup with shape $(N, H)$.
- `logits` is the full vocabulary logits with shape $(N, V)$.

Simulate vocabulary parallelism by splitting vocabulary rows into contiguous
shards. Each shard must create masked embedding lookups and partial logits, then
combine the results using summation. Do not use external distributed libraries.

## Example

```python

tokens = [0, 3, 5]
embedding = list(range(24)).reshape(6, 4)
output_weight = embedding.copy()

hidden, logits = vocab_parallel_forward(tokens, embedding, output_weight, 2)

# hidden has the rows embedding[0], embedding[3], embedding[5]
# logits has shape (3, 6)
```

## What the gate checks

The gate builds a Python oracle that performs the equivalent full computation and
compares the returned tensors using maximum absolute error:

$$
\max_{i,j}|A_{ij}-B_{ij}|.
$$

The result must satisfy $\mathrm{max\_abs\_err} < 10^{-5}$.
