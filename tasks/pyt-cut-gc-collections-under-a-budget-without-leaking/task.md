## Context

CPython frees reference cycles with a generational collector. Objects it tracks live in
three generations, and a collection of generation 0 runs whenever the number of
allocations minus deallocations since the last one exceeds a threshold — 700 by default.
A loop that builds temporary cyclic objects therefore pays for repeated scans of the young
generation, and each scan also has to walk everything promoted into the older generations.

Two controls change that. `gc.set_threshold()` raises the allocation count that triggers a
collection, and `gc.freeze()` moves everything currently tracked into a *permanent*
generation that ordinary collections do not scan at all — which is why it is used right
after interpreter start-up, before forking, so that the parent's warm heap is never walked
again.

Both controls are global and both can leak. Raise the thresholds and never collect, and the
cycles stay allocated. Freeze and never `unfreeze()`, and the objects in the permanent
generation are never reclaimed for the life of the process, because `gc.collect()` skips it.

## Task

Implement `cut_gc_collections_under_budget()`:

```python
def cut_gc_collections_under_budget(n_cycles):
    ...
```

Allocate `n_cycles` temporary reference cycles — each a pair of lists that reference each
other, appended to a list which is then cleared — with automatic collection budgeted away,
then reclaim them explicitly. Return:

- `collections`: how many garbage-collection `"stop"` callbacks fired, counted with a
  callback appended to `gc.callbacks`.
- `freed`: the value the final explicit `gc.collect()` returned, which is how many
  unreachable objects that call released.

Restore the collector's global state before returning — thresholds and enabled/disabled —
and remove your callback.

## Example

```python
collections, freed = cut_gc_collections_under_budget(200)

# collections is a small integer: only the collections you asked for
# freed accounts for the cycles the workload allocated
```

## What the gate checks

`exact_match` compares your pair against an oracle that performs the same workload and
measures it the same way, at three different sizes chosen by the grader. Returning a
memorised pair fails, because `freed` depends on `n_cycles`.

The exact value of `freed` is the interesting part, and it is not `2 * n_cycles`. It is two
lower, because after the loop the variables holding the last pair are **still bound**, which
keeps that cycle reachable — deleting them before collecting gives exactly `2 * n_cycles`.
If your number is off by two, that is where to look.

`restored_state` checks that you left the collector enabled and its thresholds not still
pinned at the budgeted values. Returning the right numbers with the collector switched off
would change how every later task in the same process behaves, so it is a wrong answer.

`gc.unfreeze()` before the final collect is good hygiene but does not change these numbers:
`freeze()` only moves objects that were *already* tracked, and the cycles you allocate
afterwards land in generation 0, which `gc.collect()` scans either way. What freezing buys is
that the scans do not have to walk the heap that existed before.

Note what is *not* measured, and why. An earlier version of this task compared
`len(gc.get_objects())` before and after the workload as a leak indicator. That counts
every tracked object in the interpreter, not yours: once the grading process had imported
the count moved for unrelated reasons, and the reference solution failed its own
task depending on who called it. The return value of `gc.collect()` is a property of your
own allocations and measures the same thing — that the cycles were actually reclaimed —
without depending on the host.
