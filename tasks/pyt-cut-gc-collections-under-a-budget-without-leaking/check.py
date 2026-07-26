"""Grade the budgeted-collection sequence on two numbers that belong to the learner's
own code, at sizes chosen here so the pair cannot be hardcoded.

Two defects in the first version of this task are worth recording, because both were
invisible until something measured them.

It compared `len(gc.get_objects())` before and after the workload as a leak indicator.
That counts every tracked object in the *interpreter*, not the learner's: once the
grading process had imported numpy the count moved for unrelated reasons, and the
reference returned `(2, False)` under `python -m mlsys grade` while returning
`(2, True)` when called directly. The reference failed its own task depending on who
called it. `gc.collect()`'s return value measures the same thing — that the cycles were
reclaimed — as a property of the submission alone: measured at 398 for 200 cycles in a
bare interpreter and 398 with numpy and the whole grader loaded.

Its statement also required `leak_flag` to be False while its own reference returned
True, so the task was self-contradictory as well as unstable.
"""
import gc

# Several sizes, so returning a memorised pair cannot pass. freed is 2*n - 2 at each of
# them, and the -2 is the point of the exercise rather than an accident: see task.md.
SIZES = (137, 200, 313)


def _run(n):
    """The oracle: the workload, with automatic collection budgeted away."""
    old_thresholds = gc.get_threshold()
    old_enabled = gc.isenabled()
    events = []

    def callback(phase, info):
        if phase == "stop":
            events.append(info.get("generation", -1))

    gc.callbacks.append(callback)
    try:
        gc.collect()

        gc.disable()
        gc.freeze()
        gc.set_threshold(1000000, 1000000, 1000000)

        roots = []
        for _ in range(n):
            a = []
            b = [a]
            a.append(b)
            roots.append(a)
        roots.clear()

        gc.unfreeze()
        freed = gc.collect()
        return len(events), freed
    finally:
        if callback in gc.callbacks:
            gc.callbacks.remove(callback)
        gc.set_threshold(*old_thresholds)
        if old_enabled:
            gc.enable()
        else:
            gc.disable()


def grade(sol, fx) -> dict:
    matches = 0
    observed = 0          # collections the submission actually caused, counted from here

    def spy(phase, info):
        nonlocal observed
        if phase == "stop":
            observed += 1

    for n in SIZES:
        expected = _run(n)
        gc.callbacks.append(spy)
        try:
            got = sol.cut_gc_collections_under_budget(n)
            got = (int(got[0]), int(got[1]))
        except Exception:
            got = None
        finally:
            if spy in gc.callbacks:
                gc.callbacks.remove(spy)
        if got == expected:
            matches += 1

    # Leaving the collector disabled or the thresholds pinned would change how every
    # task graded after this one in the same process behaves, so it is part of the answer.
    restored = 1.0 if (gc.isenabled()
                       and gc.get_threshold() != (1000000, 1000000, 1000000)) else 0.0

    # `freed` is a deterministic function of n_cycles (2*n - 2), so returning the formula
    # without touching the collector produces the right pair. Counting the collections the
    # submission causes, from outside it, is what separates doing the work from deriving
    # the answer: the workload must run the collector at least once per size.
    did_work = 1.0 if observed >= len(SIZES) else 0.0
    return {"exact_match": matches / len(SIZES),
            "restored_state": restored,
            "did_work": did_work}
