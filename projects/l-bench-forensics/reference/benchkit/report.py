from .decay import decay_table, slope_per_1k
from .parse import derive, kind
from .stats import median, separable, spread


def prefill_decode_split(rows):
    pre = [derive(r) for r in rows if kind(r) == "prefill"]
    dec = [derive(r) for r in rows if kind(r) == "decode"]
    return {
        "prefill_rows": len(pre),
        "decode_rows": len(dec),
        "best_prefill_ts": max((p["tokens_per_second"] for p in pre), default=0.0),
        "best_decode_ts": max((d["tokens_per_second"] for d in dec), default=0.0),
        "prefill_decode_ratio": (max((p["tokens_per_second"] for p in pre), default=0.0)
                                 / max((d["tokens_per_second"] for d in dec), default=1.0)),
    }


def noisiest(rows, limit=3):
    scored = []
    for r in rows:
        d = derive(r)
        if len(d["samples_ts"]) < 2:
            continue
        scored.append((spread(d["samples_ts"]), d))
    scored.sort(key=lambda x: -x[0])
    return [{"source": d["source"], "row": d["row"], "kind": d["kind"],
             "relative_iqr": s} for s, d in scored[:limit]]


def pick_ubatch(rows, min_decode_ts=0.0):
    """The micro-batch with the best prefill throughput that still meets the
    decode floor, with the evidence for the choice."""
    by_ub = {}
    for r in rows:
        ub = int(r.get("n_ubatch", 0))
        d = derive(r)
        slot = by_ub.setdefault(ub, {"prefill": [], "decode": []})
        if d["kind"] in slot:
            slot[d["kind"]].append(d)
    options = []
    for ub, slot in sorted(by_ub.items()):
        if not slot["prefill"]:
            continue
        pre = max(x["tokens_per_second"] for x in slot["prefill"])
        dec = (max(x["tokens_per_second"] for x in slot["decode"])
               if slot["decode"] else 0.0)
        options.append({"ubatch": ub, "prefill_ts": pre, "decode_ts": dec,
                        "meets_floor": dec >= min_decode_ts})
    eligible = [o for o in options if o["meets_floor"]] or options
    best = max(eligible, key=lambda o: o["prefill_ts"]) if eligible else None
    return {"options": options, "chosen": best["ubatch"] if best else None}


def model_summary(rows):
    out = {}
    for r in rows:
        name = r.get("model_type", "")
        d = derive(r)
        slot = out.setdefault(name, {
            "model": name,
            "size_bytes": r.get("model_size", 0),
            "params": r.get("model_n_params", 0),
            "best_prefill_ts": 0.0, "best_decode_ts": 0.0})
        if d["kind"] == "prefill":
            slot["best_prefill_ts"] = max(slot["best_prefill_ts"], d["tokens_per_second"])
        elif d["kind"] == "decode":
            slot["best_decode_ts"] = max(slot["best_decode_ts"], d["tokens_per_second"])
    for slot in out.values():
        slot["bytes_per_second_decode"] = slot["size_bytes"] * slot["best_decode_ts"]
        slot["params_per_second_decode"] = slot["params"] * slot["best_decode_ts"]
    return out
