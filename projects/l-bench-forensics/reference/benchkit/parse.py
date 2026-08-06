import json
import os

RESULT_FIELDS = {"avg_ns", "stddev_ns", "avg_ts", "stddev_ts", "samples_ns",
                 "samples_ts", "test_time"}
BUILD_FIELDS = {"build_commit", "build_number"}


def load(path):
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    for i, r in enumerate(rows):
        r["_source"] = os.path.basename(path)
        r["_row"] = i
    return rows


def load_all(paths):
    out = []
    for p in paths:
        out.extend(load(p))
    return out


def kind(row):
    if row.get("n_prompt", 0) and not row.get("n_gen", 0):
        return "prefill"
    if row.get("n_gen", 0) and not row.get("n_prompt", 0):
        return "decode"
    return "mixed"


def tokens(row):
    return int(row.get("n_prompt", 0)) + int(row.get("n_gen", 0))


def derive(row):
    n = tokens(row)
    avg_s = row["avg_ns"] / 1e9
    return {
        "source": row["_source"],
        "row": row["_row"],
        "kind": kind(row),
        "tokens": n,
        "depth": int(row.get("n_depth", 0)),
        "ubatch": int(row.get("n_ubatch", 0)),
        "batch": int(row.get("n_batch", 0)),
        "model": row.get("model_type", ""),
        "avg_seconds": avg_s,
        "tokens_per_second": n / avg_s if avg_s else 0.0,
        "ms_per_token": avg_s * 1000.0 / n if n else 0.0,
        "reported_ts": row["avg_ts"],
        "samples_ts": list(row.get("samples_ts") or []),
        "reps": len(row.get("samples_ns") or []),
    }


def config(row):
    """Everything that describes how the run was set up, results excluded."""
    return {k: v for k, v in row.items()
            if not k.startswith("_") and k not in RESULT_FIELDS
            and k not in BUILD_FIELDS and not isinstance(v, list)}
