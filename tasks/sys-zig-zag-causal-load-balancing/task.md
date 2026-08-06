## Context

Context parallelism splits one long sequence across $p$ ranks. Under a causal
mask, chunk $i$ attends to chunks $0, 1, \dots, i$, so its work weight is

$$
w_i = i + 1 .
$$

The obvious split — one contiguous chunk per rank — is badly imbalanced: the
last rank does $p$ times the work of the first, and at every step boundary the
other $p-1$ ranks wait for it.

Zig-zag (striped) assignment fixes this by cutting the sequence into $2p$ chunks
instead of $p$, then giving each rank one early chunk and one late chunk. The
two weights are complementary, so every rank ends up with exactly the same
causal work.

## Task

Implement `zigzag_assignment(num_ranks)`:

```python
def zigzag_assignment(num_ranks: int) -> list[int]:
    ...
```

The sequence is cut into $2 \cdot \texttt{num\_ranks}$ chunks. Return a list of
length $2 \cdot \texttt{num\_ranks}$ whose element $i$ is the rank that owns
chunk $i$.

The assignment must satisfy all three:

1. Every rank owns exactly two chunks.
2. Rank $r$ owns chunk $r$ — the first $p$ chunks go to ranks $0 \dots p-1$ in
   order.
3. The per-rank causal work $\sum_{i \in \text{rank } r} (i+1)$ is the same for
   every rank.

Ranks are numbered $0$ to `num_ranks - 1`. `num_ranks` is at least $1$.

## Example

```python
zigzag_assignment(4)
```

```text
[0, 1, 2, 3, 3, 2, 1, 0]
```

Rank $0$ owns chunks $0$ and $7$: work $1 + 8 = 9$. Rank $1$ owns chunks $1$ and
$6$: $2 + 7 = 9$. Every rank gets $9$.

A contiguous split of the same $8$ chunks into $4$ consecutive pairs gives
per-rank work $3, 7, 11, 15$ — the last rank does $5\times$ the first.

## What the gate checks

Two metrics.

`exact_match` is $1.0$ only when the returned list equals the oracle assignment
for every tested `num_ranks`: correct length, rank $r$ on chunk $r$, and the
matching pairing in the second half.

`imbalance` is computed by the gate from **your** assignment — it is not a
number you return:

$$
\text{imbalance} =
\frac{\max_r \sum_{i \in \text{rank } r}(i+1)}
     {\min_r \sum_{i \in \text{rank } r}(i+1)} .
$$

It must be exactly $1.0$.

A contiguous split scores `exact_match` $=0$ and `imbalance` $=5.0$ at $p=4$.
Balancing the *chunk count* instead of the *causal work* also fails: two chunks
per rank is necessary but not sufficient.
