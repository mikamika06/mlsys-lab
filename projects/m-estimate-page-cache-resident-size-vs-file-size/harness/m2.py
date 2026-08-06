import numpy as np
import ref


def check(workdir):
    from pagecache.tracker import PageCacheTracker

    out = {"rel_err": 0.0}
    total_err = 0.0
    count = 0

    for fixture in ref.FIXTURES_M2:
        want_history = ref.simulate_cache_tracker(
            fixture["file_size"],
            fixture["read_events"],
            fixture["evict_pages"],
        )

        tracker = PageCacheTracker(fixture["file_size"])
        got_history = []
        for accesses in fixture["read_events"]:
            tracker.access(accesses)
            got_history.append(tracker.get_resident_bytes())

        tracker.evict(fixture["evict_pages"])
        got_history.append(tracker.get_resident_bytes())

        for w, g in zip(want_history, got_history):
            err = abs(g - w) / float(w) if w > 0 else 0.0
            total_err += err
            count += 1

    out["rel_err"] = float(total_err / count) if count > 0 else 0.0
    return out
