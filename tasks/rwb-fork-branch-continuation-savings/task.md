## Context

Agent/chat systems commonly **fork**: a shared trunk of context (system
prompt, retrieved documents, conversation so far) is used to spawn $K$
independent continuations — parallel tool calls, sampled alternatives, or
separate branches of a conversation tree. A production prefix-cache
(RadixAttention, vLLM APC, ...) stores every generated sequence in one
shared tree keyed by tokens, so that inserting a new sequence only pays
for the tokens that are genuinely new — any prefix it shares with
something already in the tree is free.

For a set of full sequences $s_1, \ldots, s_K$ inserted **in order**, the
number of tokens *saved* by sequence $s_i$ is the length of the longest
prefix of $s_i$ that already exists in the tree from an earlier insert:

$$
\text{saved}_i = \max \{ p \;:\; s_i[0{:}p] \text{ is already a path in the tree after inserting } s_1, \ldots, s_{i-1} \}
$$

If every branch shares only the trunk $T$ and diverges immediately after
it, $\text{saved}_i = |T|$ for every branch except the first
($\text{saved}_1 = 0$, since nothing exists yet). But branches can also
happen to share structure *beyond* the trunk — e.g. two tool-call
continuations that both start by re-stating the same function signature —
and a real tree captures that too.

## Task

Implement `branch_savings(trunk, continuations)`:

```python
def branch_savings(trunk: list[int], continuations: list[list[int]]) -> int:
    ...
```

- `trunk`: the shared prefix token list.
- `continuations`: a list of $K$ token lists; branch $i$'s full sequence
  is `trunk + continuations[i]`.

Insert the $K$ full sequences into one shared prefix tree, **in the given
order**, and return the total number of tokens saved (summed over all $K$
branches) by reuse of anything already in the tree.

## Example

```python
trunk = [1, 2, 3, 4, 5]
continuations = [[10, 11], [20, 21, 22], [30], [40, 41, 42, 43]]
branch_savings(trunk, continuations)
# branch 0: nothing cached yet -> saved 0, tree now holds trunk+[10,11]
# branch 1: trunk (5 tokens) already present, then diverges at 20 -> saved 5
# branch 2: trunk already present, diverges at 30 -> saved 5
# branch 3: trunk already present, diverges at 40 -> saved 5
# -> 0 + 5 + 5 + 5 = 15
```

## What the gate checks

The gate runs several hand-built scenarios (a clean trunk-plus-divergent-
branches case like the example, an empty trunk, branches that additionally
share a sub-prefix *beyond* the trunk before diverging further, an exact
duplicate branch, and a trunk with no continuations at all) plus several
randomly generated `(trunk, continuations)` cases from a seeded generator
that sometimes gives multiple branches an extra shared segment right after
the trunk.

For every case the reference inserts the same sequences into a real
token-by-token prefix tree, in the same order, and sums up exactly how
many tokens each insert found already present. Your return value is
compared to that total with exact equality. A solution that only ever
credits the fixed trunk length (e.g. `len(trunk) * (K - 1)`) and ignores
both (a) that the very first branch pays for the trunk instead of saving
it, and (b) any additional structure branches happen to share beyond the
trunk, will disagree with the oracle on the mixed-scenario cases.
