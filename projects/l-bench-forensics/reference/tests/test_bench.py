from benchkit.decay import decay_table
from benchkit.parse import derive, kind
from benchkit.stats import spread


def audit(rows):
    warnings = []
    models = sorted({r.get("model_type", "") for r in rows})
    for model in models:
        table = decay_table(rows, model=model)
        for entry in table:
            if entry["loss_fraction"] < 0 and entry["separable_from_empty"]:
                warnings.append(
                    "%s: depth %d is faster than depth 0 by %.1f%%, and the "
                    "spreads do not overlap; the empty-context row is suspect"
                    % (model, entry["depth"], -entry["loss_fraction"] * 100))
        for a, b in zip(table, table[1:]):
            if b["tokens_per_second"] > a["tokens_per_second"] and a["depth"] == 0:
                continue
            if b["tokens_per_second"] > a["tokens_per_second"]:
                warnings.append(
                    "%s: throughput rises from depth %d to depth %d, which the "
                    "cache cannot explain" % (model, a["depth"], b["depth"]))
    for r in rows:
        d = derive(r)
        if len(d["samples_ts"]) >= 2 and spread(d["samples_ts"]) > 0.1:
            warnings.append(
                "%s row %d (%s, depth %d): inter-quartile spread is %.0f%% of the "
                "median; the average is not worth quoting"
                % (d["source"], d["row"], kind(r), d["depth"],
                   spread(d["samples_ts"]) * 100))
    return warnings
