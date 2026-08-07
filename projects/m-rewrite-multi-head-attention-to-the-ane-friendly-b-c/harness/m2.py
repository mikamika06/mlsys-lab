import ref


def check(workdir):
    from aneattn import convert, metrics

    out = {"latency_improved": 0.0, "ops_reduced": 0.0}

    try:
        naive_lat = convert.measure_latency("naive")
        ane_lat = convert.measure_latency("ane")

        if ane_lat < naive_lat:
            out["latency_improved"] = 1.0
        else:
            out["_note"] = f"latency not improved: naive {naive_lat}, ane {ane_lat}"

        naive_ops = metrics.count_ops("naive")
        ane_ops = metrics.count_ops("ane")

        if (ane_ops["reshape"] < naive_ops["reshape"]) and (ane_ops["transpose"] < naive_ops["transpose"]):
            out["ops_reduced"] = 1.0
        else:
            out["_note"] = f"ops not reduced properly: naive {naive_ops}, ane {ane_ops}"
    except Exception as e:
        out["_note"] = f"exception in convert/metrics: {str(e)[:100]}"

    return out
