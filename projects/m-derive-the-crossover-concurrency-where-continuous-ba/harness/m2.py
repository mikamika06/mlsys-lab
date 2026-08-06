import ref
import numpy as np

def check(workdir):
    from crossover.normalize import normalize_benchmarks
    benchmarks = ref.get_benchmarks()
    want = [ref.normalize_benchmark(b) for b in benchmarks]
    got = normalize_benchmarks(benchmarks)
    if not isinstance(got, (list, tuple)) or len(got) != len(want):
        return {"rel_err": 1.0}
    errs = [abs(g - w) / max(1.0, abs(w)) for g, w in zip(got, want)]
    mean_err = float(np.mean(errs))
    return {"rel_err": mean_err}
