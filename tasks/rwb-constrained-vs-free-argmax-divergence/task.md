## Context

Grammar/structured-output decoders (outlines, xgrammar, lm-format-enforcer)
drive an FSM alongside the model: at each generation step $t$, the FSM's
current state $s_t$ determines the set of tokens $A(s_t) \subseteq
\{0, \ldots, V-1\}$ that are legal to emit next (the rest would produce a
string the grammar rejects). In practice this is implemented as a **logit
mask**: build a vector that is $0$ where a token is allowed and $-\infty$
elsewhere, add it to the raw logits, and take the argmax of the result —

$$
\text{constrained\_argmax}(t) = \arg\max_{v \in A(s_t)} \; \text{logits}_t[v]
$$

$$
\text{free\_argmax}(t) = \arg\max_{v \in \{0,\ldots,V-1\}} \; \text{logits}_t[v]
$$

Most of the time these two agree — the model's own top pick already
happens to satisfy the grammar. The steps where they **disagree** are the
steps where the grammar constraint actually changed the model's output;
counting them tells you how often the constraint is doing real work versus
just being satisfied for free.

## Task

Implement `constrained_free_argmax_divergence(logits, trace, allowed)`:

```python
def constrained_free_argmax_divergence(logits, trace, allowed) -> int:
    ...
```

- `logits`: `(T, vocab_size)` array of per-step logits.
- `trace`: length-`T` list of FSM state ids, one per decoding step.
- `allowed`: `dict[state_id -> list[int]]`, the set of allowed token ids
  for each FSM state.

For every step $t$ (using `state = trace[t]`), compute the free argmax
(over the whole vocab) and the constrained argmax (over `allowed[state]`
only, ties broken by the lowest token id — i.e. mask disallowed tokens to
$-\infty$ and take `argmax` of the result, exactly like a real masked
decode). Return the **count** of steps where the two differ.

## Example

```python
logits = np.array([
    [5.0, 1.0, 2.0, 0.0],   # free argmax = 0
    [1.0, 5.0, 2.0, 0.0],   # free argmax = 1
    [1.0, 2.0, 5.0, 0.0],   # free argmax = 2
])
trace = [0, 1, 1]
allowed = {0: [0, 1], 1: [0, 2]}

constrained_free_argmax_divergence(logits, trace, allowed)
# step 0: state 0, allowed={0,1}, constrained argmax = 0 -> matches free (0)
# step 1: state 1, allowed={0,2}, constrained argmax = 0 -> free was 1: DIVERGES
# step 2: state 1, allowed={0,2}, constrained argmax = 2 -> matches free (2)
# -> 1
```

## What the gate checks

The gate covers three hand-built scenarios (an FSM state whose allowed set
is the entire vocab, so it never diverges; an allowed set pinned to a
single token, so it diverges on every step whose free argmax isn't that
token; and a small hand-crafted logits/mask table mixing FSM states with
and without the top-1 token allowed) plus several randomly generated
`(logits, trace, allowed)` triples from a seeded generator with varying
vocab size, number of FSM states, and allowed-set sizes.

For each case the reference recomputes both argmax sequences with NumPy
exactly as described above — masking disallowed tokens to $-\infty$ before
taking `argmax`, so ties resolve to the lowest token id on both the free
and the constrained side — and counts the disagreements. Your return value
is compared to that count with exact equality on every case. A solution
that looks up the allowed set once (e.g. from `trace[0]`) and reuses it
for every step, instead of re-reading `allowed[trace[t]]` fresh at each
step, will match on scenarios where the FSM stays in one state the whole
time but silently diverge from the oracle as soon as a test case switches
states mid-sequence.
