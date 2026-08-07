import importlib.util
import os
import sys


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))]
    if not fns:
        return None
    for fn in fns:
        fn()
    return True


def _survives(path):
    try:
        return _run(path) is True
    except Exception:
        return False


def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_min_seq_len_bug": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")

    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"the tests fail on a correct implementation: {e}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import serving.queue as q
    good_simulate = q.simulate

    def broken_simulate(arrivals, seq_lens, max_batch, timeout):
        from serving.models import profile_latency
        n = len(arrivals)
        latencies = []
        current_time = 0.0
        idx = 0
        queue = []

        while idx < n or queue:
            if not queue and idx < n and current_time < arrivals[idx]:
                current_time = arrivals[idx]

            while idx < n and arrivals[idx] <= current_time:
                queue.append((arrivals[idx], seq_lens[idx]))
                idx += 1

            if queue:
                wait_time = current_time - queue[0][0]
                if len(queue) >= max_batch or wait_time >= timeout - 1e-9:
                    batch = queue[:max_batch]
                    queue = queue[max_batch:]
                    b_size = len(batch)
                    b_seq = min(item[1] for item in batch)
                    exec_time = profile_latency(b_size, b_seq)

                    finish_time = current_time + exec_time
                    for arr, seq in batch:
                        latencies.append(finish_time - arr)
                    current_time = finish_time
                else:
                    next_timeout = queue[0][0] + timeout
                    if idx < n:
                        current_time = min(arrivals[idx], next_timeout)
                    else:
                        current_time = next_timeout

        s_lat = sorted(latencies)
        return {
            "p50": s_lat[int(len(s_lat) * 0.50)],
            "p99": s_lat[int(len(s_lat) * 0.99)]
        }

    q.simulate = broken_simulate
    try:
        if not _survives(path):
            out["catches_min_seq_len_bug"] = 1.0
        else:
            out["_note"] = "test passed even when simulation incorrectly batched by minimum sequence length"
    finally:
        q.simulate = good_simulate

    return out
