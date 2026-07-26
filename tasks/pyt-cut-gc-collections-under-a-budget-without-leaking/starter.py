import gc


def cut_gc_collections_under_budget(n_cycles):
    """Allocate n_cycles temporary reference cycles with automatic collection
    budgeted away, then reclaim them explicitly.

    Return (collections, freed):
      collections — how many gc "stop" callbacks fired while your workload ran
      freed       — what the final explicit gc.collect() returned

    Restore the collector's global state before returning, and remove your callback.
    """
    raise NotImplementedError('your code here')
