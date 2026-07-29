import importlib.util
import os


def _load_measure(workdir):
    path = os.path.join(workdir, "bench", "bench.py")
    spec = importlib.util.spec_from_file_location("learner_bench", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.measure


def check(workdir):
    out = {"warmup_untimed": 0.0, "reports_median": 0.0, "reps_honoured": 0.0,
           "syncs_after_each_rep": 0.0, "reports_spread": 0.0}
    measure = _load_measure(workdir)

    ticks = {"n": 0}
    calls = {"n": 0}
    syncs = {"n": 0}

    def timer():
        ticks["n"] += 1
        return float(ticks["n"])

    def fn():
        calls["n"] += 1

    def sync():
        syncs["n"] += 1

    r = measure(fn, warmup=3, reps=5, timer=timer, sync=sync)
    out["reps_honoured"] = 1.0 if calls["n"] == 8 else 0.0
    out["warmup_untimed"] = 1.0 if r.get("reps", 5) == 5 else 0.0
    out["syncs_after_each_rep"] = 1.0 if syncs["n"] >= 5 else 0.0

    seq = [0.0, 1.0, 1.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 100.0]
    idx = {"i": 0}

    def skewed():
        v = seq[min(idx["i"], len(seq) - 1)]
        idx["i"] += 1
        return v

    def noop():
        pass

    r2 = measure(noop, warmup=0, reps=5, timer=skewed, sync=None)
    med = r2.get("median")
    out["reports_median"] = 1.0 if med is not None and abs(med - 1.0) < 1e-9 else 0.0
    out["reports_spread"] = 1.0 if ("iqr" in r2 or "q3" in r2) else 0.0
    if not out["reports_median"]:
        out["_note"] = f"median on a skewed sample came back as {med}, expected 1.0"
    return out
