import glob
import json
import os

FIX = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "_fixtures", "llama_bench"))

RESULT_FIELDS = {"avg_ns", "stddev_ns", "avg_ts", "stddev_ts", "samples_ns",
                 "samples_ts", "test_time"}
BUILD_FIELDS = {"build_commit", "build_number"}


def files():
    return sorted(glob.glob(os.path.join(FIX, "*.json")))


def raw():
    out = []
    for p in files():
        with open(p, encoding="utf-8") as f:
            for i, r in enumerate(json.load(f)):
                r["_source"] = os.path.basename(p)
                r["_row"] = i
                out.append(r)
    return out


def _med(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _q(xs):
    s = sorted(xs)
    n = len(s)
    if n < 2:
        return (s[0], s[0], s[0]) if n else (0.0, 0.0, 0.0)
    m = n // 2
    return _med(s[:m]), _med(s), _med(s[m + 1:] if n % 2 else s[m:])


def expect_derive(row):
    n = int(row.get("n_prompt", 0)) + int(row.get("n_gen", 0))
    secs = row["avg_ns"] / 1e9
    if row.get("n_prompt", 0) and not row.get("n_gen", 0):
        k = "prefill"
    elif row.get("n_gen", 0) and not row.get("n_prompt", 0):
        k = "decode"
    else:
        k = "mixed"
    return {"kind": k, "tokens": n, "tokens_per_second": n / secs,
            "ms_per_token": secs * 1000.0 / n,
            "reps": len(row.get("samples_ns") or [])}


def expect_stats(samples):
    q1, q2, q3 = _q(samples)
    return {"median": q2, "iqr": q3 - q1}


def expect_separable(a, b):
    a1, _, a3 = _q(a)
    b1, _, b3 = _q(b)
    if len(a) < 2 or len(b) < 2:
        return 0
    return 1 if (a3 < b1 or b3 < a1) else 0


def _config(row):
    return {k: v for k, v in row.items()
            if not k.startswith("_") and k not in RESULT_FIELDS
            and k not in BUILD_FIELDS and not isinstance(v, list)
            and k != "model_filename"}


def expect_differences(a, b):
    ca, cb = _config(a), _config(b)
    return sorted(k for k in set(ca) | set(cb) if ca.get(k) != cb.get(k))


def expect_decay(rows, model):
    per = {}
    for r in rows:
        if r.get("model_type") != model:
            continue
        if not (r.get("n_gen", 0) and not r.get("n_prompt", 0)):
            continue
        per[int(r.get("n_depth", 0))] = r
    if 0 not in per:
        return []
    base = _med(per[0]["samples_ts"])
    out = []
    for depth in sorted(per):
        ts = _med(per[depth]["samples_ts"])
        out.append({"depth": depth, "tokens_per_second": ts,
                    "loss_fraction": 1.0 - ts / base,
                    "separable_from_empty": expect_separable(
                        per[depth]["samples_ts"], per[0]["samples_ts"])})
    return out


def models():
    return sorted({r.get("model_type", "") for r in raw()})


def near(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(b))
