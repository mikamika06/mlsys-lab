## Context

An open-addressing hash table stores every entry directly in a fixed-size
slot array (no linked buckets). A key $k$ starts its probe at slot
$h(k) \bmod C$ (capacity $C$) and, on collision, walks forward
$h(k)+1, h(k)+2, \dots \pmod C$ until it finds either the key itself or a
truly empty slot.

Deletion is where naive implementations break. If `delete` simply resets a
slot to "empty", it silently **breaks every probe chain that walked through
it**. Concretely: insert keys $k_1, k_2, k_3$ that all hash to the same slot
$s$, so $k_1 \to s$, $k_2 \to s{+}1$, $k_3 \to s{+}2$. Delete $k_2$ by
blanking slot $s{+}1$. Now look up $k_3$: the probe starts at $s$ (occupied by
$k_1$, not a match), steps to $s{+}1$ — finds it **empty** — and stops,
reporting `k3` as missing even though it is still sitting at $s{+}2$.

The fix is a **tombstone** (a `DELETED` sentinel, distinct from `EMPTY`):
probing must *skip over* tombstones (they don't end the search) but *stop at*
a real `EMPTY` slot (that does end the search — nothing further down the
chain was ever placed). Insertion is free to reuse a tombstone slot.

## Task

Fix the from-scratch hash table:

```python
class ScratchDict:
    def __init__(self, capacity: int = 16):
        ...
    def set(self, key: int, value) -> None:
        ...
    def get(self, key: int):
        """Return the value for key, or raise KeyError if absent."""
        ...
    def delete(self, key: int) -> None:
        """Remove key, or raise KeyError if absent."""
        ...
    def __contains__(self, key: int) -> bool:
        ...
```

* Keys are non-negative Python `int`s; hash slot $= \texttt{hash(key)} \bmod
  \texttt{capacity}$ (for `int`, `hash(k) == k`), then **linear probing**
  (step $+1 \bmod$ capacity) on collision.
* `capacity` never needs to grow in this task — assume it is always large
  enough for the operations performed.
* The provided implementation blanks a slot straight to `EMPTY` on delete,
  which breaks lookups the way described above. Fix `delete` (and, if needed,
  the probe loop in `get`/`set`/`__contains__`) so probing correctly skips
  tombstones but still stops at a truly empty slot, and so a later `set` can
  reuse a tombstoned slot.

## Example

```python
d = ScratchDict(capacity=8)
d.set(3, "a")     # slot 3
d.set(11, "b")    # 11 % 8 == 3, collides -> slot 4
d.set(19, "c")    # 19 % 8 == 3, collides -> slot 5
d.delete(11)      # tombstone at slot 4
d.get(19)         # must still return "c" -- probe must not stop at slot 4
11 in d           # False
```

## What the gate checks

The grader runs a long, fixed sequence of `set` / `delete` / `get` /
`__contains__` calls — built around deliberately colliding integer keys
(several keys sharing one hash bucket, interior deletions, reinsertion into a
tombstoned slot, consecutive tombstones, deleting the *first* key in a chain)
— against **both** your `ScratchDict` and a real Python `dict` used as the
oracle. After every operation the grader compares your `get`/`__contains__`
result (including whether a `KeyError` was correctly raised for an absent
key) to what the real `dict` says. `exact_match` is `1.0` only if every
single operation in the sequence agrees with the oracle, `0.0` on the first
disagreement.
