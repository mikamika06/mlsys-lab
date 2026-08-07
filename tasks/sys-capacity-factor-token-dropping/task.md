## Context

In a Mixture-of-Experts (MoE) layer each input token is routed to one or more
specialist sub-networks (experts). To bound per-expert computation and memory,
every expert has a **capacity** — the maximum number of tokens it will process
in one forward pass. Tokens that exceed an expert's capacity are **dropped**
(masked out) and do not contribute to that expert's computation.

The per-expert capacity is set by a **capacity factor** $C \geq 0$:

$$c = \left\lceil \frac{C \cdot N}{E} \right\rceil$$

where $N$ is the total token count and $E$ is the number of experts. When
$C \geq 1$ and routing is roughly uniform, every token is kept. When $C < 1$,
some tokens must be dropped. The capacity factor trades computational budget
against information loss from dropped tokens.

Formally, let $\text{assign}(i) \in \{0, \dots, E-1\}$ denote the expert
assigned to token $i$. For each expert $j$, define the set of its assigned
tokens as

$$S_j = \{\, i \mid \text{assign}(i) = j \,\}$$

sorted by index. Expert $j$ keeps the first $\min(|S_j|, c)$ tokens in $S_j$
and drops the rest. A token $i$ is **kept** iff it survives this filtering
for its assigned expert.

## Task

Implement `token_drop_mask`:

```python
def token_drop_mask(assignments, num_experts, capacity_factor):
```

**Parameters:**

- `assignments` — a 1-D integer list of shape $(N,)$.
  `assignments[i]` is the 0-indexed expert that token $i$ is routed to.
  Every value satisfies $0 \le \text{assignments}[i] < E$.
- `num_experts` — the number of experts $E \ge 1$.
- `capacity_factor` — the capacity factor $C \ge 0$ (may be fractional).

**Returns:**

A 1-D boolean list of shape $(N,)$ where `True` means the token is
**kept** and `False` means it is **dropped**.

**Dropping rule:** Compute $c = \lceil C \cdot N / E \rceil$. For each
expert $j$ collect all token indices $i$ with $\text{assign}(i) = j$ in
ascending order. Keep the first $\min(|S_j|,\; c)$ and drop the remainder.

## Example

```python

assignments = [0, 1, 0, 1, 0]   # N = 5 tokens, E = 2 experts
mask = token_drop_mask(assignments, num_experts=2, capacity_factor=0.8)
# c = ceil(0.8 * 5 / 2) = ceil(2.0) = 2
# Expert 0: indices [0, 2, 4] → keep 2 → tokens 0, 2 kept; 4 dropped
# Expert 1: indices [1, 3]   → keep 2 → tokens 1, 3 kept
# mask == [True, True, True, True, False]
```

## What the gate checks

The gate uses `exact_match`: your boolean mask must agree element-by-element
with a reference mask across every test case. The reference is computed from
the oracle algorithm described above (`math.ceil` for capacity, earliest
indices kept per expert), so there is no ambiguity in the expected output.
Any deviation — wrong capacity formula, random / priority-based dropping, or
off-by-one in the keep count — causes a mismatch.
