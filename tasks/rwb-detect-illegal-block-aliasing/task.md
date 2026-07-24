## Context

In a paged KV cache, several live sequences hold **block tables**
mapping their logical blocks to physical blocks in a shared pool. Prefix
sharing (radix-cache reuse, automatic prefix caching) lets two sequences
point at the *same* physical block **read-only** — a copy-on-write (COW)
share, marked `is_shared = True` on both sides. That is safe: neither
sequence writes through that mapping, so nothing is corrupted.

It becomes **illegal aliasing** the moment a physical block is
referenced by two or more distinct live sequences and *at least one* of
those references is a private, writable mapping (`is_shared = False`).
A write through that private mapping would silently corrupt whatever the
other sequence(s) believe is still their own (or their shared) data.
Formally, for physical block $p$ referenced by the set of (sequence,
`is_shared`) pairs $R_p$:

$$
\text{legal}(p) \iff |R_p| \le 1 \ \lor\ \forall (\cdot, s) \in R_p:\ s = \text{True}
$$

A whole configuration is legal iff `legal(p)` holds for every physical
block $p$ referenced by any sequence.

## Task

Implement `is_block_mapping_legal`:

```python
def is_block_mapping_legal(seqs: list[dict]) -> bool:
    ...
```

- `seqs`: a list of live sequences, each a dict
  `{"physical_block_ids": list[int], "is_shared": list[bool]}` of equal
  length per sequence — `physical_block_ids[i]` is the physical block
  that sequence's logical block `i` maps to, `is_shared[i]` says whether
  that mapping is read-only-shared (`True`) or private/writable
  (`False`) **for that sequence**.
- A sequence referencing the same physical block more than once within
  itself is not aliasing — only cross-sequence references matter.
- Return `True` iff no physical block violates the legality condition
  above.

## Example

```python
seqs = [
    {"physical_block_ids": [0, 1, 5], "is_shared": [True, True, False]},
    {"physical_block_ids": [0, 1, 6], "is_shared": [True, True, False]},
]
is_block_mapping_legal(seqs)  # True -- blocks 0, 1 are a consistent shared prefix

seqs[1]["is_shared"][0] = False   # sequence 1 now treats block 0 as private
is_block_mapping_legal(seqs)  # False -- block 0 is aliased with a writable mapping
```

## What the gate checks

The grader runs 8 handcrafted edge cases (disjoint sequences, a
consistent multi-way shared prefix, the same prefix with one sequence's
flag flipped, two private sequences accidentally colliding on one block,
a sequence referencing its own block twice, and a lone sequence) plus 8
seeded random configurations, and classifies each independently with an
oracle that re-derives legality directly from the definition above —
never calling your function, never hardcoding an expected label.

`exact_match` is the fraction of the 16 configurations your function
classifies correctly and must equal `1.0`. Treating any multi-sequence
reference as automatically legal (or illegal), requiring *all* sharers
instead of checking every individual flag, or double-counting a
sequence's internal repeat reference as cross-sequence aliasing will all
misclassify at least one case.
