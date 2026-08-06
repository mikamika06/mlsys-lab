import ref


def check(workdir):
    from bnb_quant.bench import measure_loading_latency, measure_memory_footprint

    out = {"latency_ratio": 0.0, "bytes_saved_ratio": 0.0}
    model = ref.TEST_MODELS[-1]

    lat_res = measure_loading_latency(model)
    mem_res = measure_memory_footprint(model)

    out["latency_ratio"] = float(lat_res["ratio"])
    out["bytes_saved_ratio"] = float(mem_res["bytes_saved_ratio"])
    return out
