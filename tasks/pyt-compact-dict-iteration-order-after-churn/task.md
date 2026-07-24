## Context

Since CPython 3.6 (guaranteed language behavior since 3.7), `dict` uses the
"compact dict" layout: a sparse table of indices (used only for hashing /
probing) plus a **dense array of (key, value) entries stored in insertion
order**. Iterating a dict walks that dense array, which is why
`list(d.keys())` always reflects insertion order — independent of the keys'
hash values.

The subtlety is what happens under churn (a mix of inserts, updates, and
deletes):

- Inserting a **new** key appends a new slot to the end of the dense array.
- **Updating** the value of a key that is *already present* does **not**
  move it — it keeps its existing position in the dense array.
- **Deleting** a key removes its slot from the dense array entirely.
- Re-**inserting** a key that was previously deleted is indistinguishable
  from inserting a brand-new key: it goes back in at the **current end**,
  not restored to wherever it used to sit.

Formally, if $S$ is the sequence of `set`/`del` operations applied to an
initially empty dict, the resulting iteration order is exactly the order of
first-occurrence-since-last-deletion among the keys still present at the
end — plain value updates are invisible to this order, but a
delete-then-reinsert resets a key's position to "now."

## Task

Implement `predict_dict_order(ops)`:

```python
def predict_dict_order(ops: list[tuple]) -> list[int]:
    ...
```

`ops` is a list of 2-tuples with integer keys:

- `("set", k)` — set/update key `k` (the value doesn't matter for order).
- `("del", k)` — delete key `k` (guaranteed to be present at that point).

Applying `ops` in order to an initially empty dict, return the list of keys
in the resulting dict's iteration order (i.e. what `list(d.keys())` would
give).

## Example

```python
ops = [("set", 1), ("set", 2), ("set", 3), ("del", 2), ("set", 2)]
predict_dict_order(ops)
# -> [1, 3, 2]
# 2 is deleted and then re-inserted, so it lands at the END, not restored
# to its old middle position.

ops2 = [("set", 1), ("set", 2), ("set", 3), ("set", 1)]
predict_dict_order(ops2)
# -> [1, 2, 3]
# key 1 already exists, so ("set", 1) is a plain value update: its
# position is unchanged.
```

## What the gate checks

The gate generates several random operation sequences with a seeded
generator (mixing fresh inserts, updates of already-present keys, and
deletes of currently-present keys so every sequence is valid) plus a couple
of hand-picked edge cases like the ones above. For each sequence it builds
the reference order by literally applying the operations to a real Python
`dict` and taking `list(d.keys())` — the real CPython dict *is* the oracle
here, since dict ordering is a documented language guarantee, not an
implementation detail to approximate.

Your `predict_dict_order` output is compared to that reference list with
exact equality (same keys, same order) on every test sequence; the metric
is `1.0` only if every sequence matches exactly, else `0.0`. A solution that
treats every `("set", k)` as moving `k` to the end — as if the dict behaved
like an LRU cache that promotes on every touch, including plain updates —
gets deletes-and-fresh-inserts right but reorders keys on ordinary updates,
and fails as soon as a test sequence updates a key that isn't already last.
