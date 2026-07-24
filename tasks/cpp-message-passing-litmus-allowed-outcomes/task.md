## Context

In concurrent C++, the **Message-Passing (MP)** litmus test asks whether
data written by a Producer thread $T_1$ is guaranteed visible to a
Consumer thread $T_2$ once it observes a notification flag. Two shared
variables, `data_field` and `flag_field`, both start at `0`:

- **Producer $T_1$**: `data_field = data_val;` then
  `flag_field.store(1, write_mo);`
- **Consumer $T_2$**: `r1 = flag_field.load(read_mo);` then
  `r2 = data_field;`

**Unsynchronized pair** (`write_mo` not one of `{release, acq_rel,
seq_cst}`, or `read_mo` not one of `{acquire, acq_rel, seq_cst}`): no
*happens-before* edge is established. $T_2$ can observe `r1 == 1` (the
flag update) while still seeing the stale `r2 == 0` (the data write not
yet visible), because nothing prevents the two writes in $T_1$ from
appearing reordered from $T_2$'s point of view.
Allowed outcomes: $\{(0,0), (0,v), (1,0), (1,v)\}$.

**Synchronized release/acquire pair** (`write_mo` $\in$ `{release,
acq_rel, seq_cst}` AND `read_mo` $\in$ `{acquire, acq_rel, seq_cst}`): the
release store synchronizes-with the acquire load. Once $T_2$ observes
`r1 == 1`, every write $T_1$ made before the store — including
`data_field = data_val` — is guaranteed visible. Outcome $(1, 0)$ is
**forbidden**.
Allowed outcomes: $\{(0,0), (0,v), (1,v)\}$.

## Task

Implement, in `solve.cpp`,

```cpp
std::vector<std::pair<int, int>> get_allowed_litmus_outcomes(
    std::memory_order write_mo, std::memory_order read_mo, int data_val);
```

Apply the rule above to decide whether `write_mo`/`read_mo` form a
synchronized release/acquire pair, then return the corresponding set of
allowed `(r1, r2)` outcomes, sorted lexicographically with no duplicates
(note `data_val` could be `0`, which would otherwise collide with the
`(0, 0)` / `(1, 0)` entries).

## Example

`get_allowed_litmus_outcomes(memory_order_release, memory_order_acquire,
42)` → `{(0,0), (0,42), (1,42)}` (synchronized: `(1,0)` is impossible).

`get_allowed_litmus_outcomes(memory_order_relaxed, memory_order_acquire,
42)` → `{(0,0), (0,42), (1,0), (1,42)}` (unsynchronized: all four
combinations are possible).

## What the gate checks

The fixed driver (`main.cpp`) runs six fixed `(write_mo, read_mo,
data_val)` cases spanning every relevant combination of synchronized and
unsynchronized memory orders, and prints every allowed outcome pair for
each. The gate is an exact string match (`exact_match == 1.0`) against
the reference's printed output: misclassifying even one memory-order
combination, or missing/adding a single outcome, changes the printed set
and fails the gate.
