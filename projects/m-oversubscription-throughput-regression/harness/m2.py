import ref


def check(workdir):
    from oversub.scheduler import OptimalStreamPool

    out = {"throughput_ratio": 0.0, "pools_valid": 0.0}
    ratios = []

    for i, sc in enumerate(ref.SCENARIOS):
        bench = ref.make_bench_fn(sc["num_cores"], sc["scaling"], sc["penalty"])
        try:
            pool = OptimalStreamPool(bench, sc["max_streams"])
            opt = pool.get_optimal_streams()
            if opt != sc["num_cores"]:
                out["_note"] = f"scenario {i}: optimal streams {opt} != expected {sc['num_cores']}"
                return out

            ratio = pool.compute_throughput_ratio(sc["max_streams"])
            ratios.append(ratio)
        except Exception as e:  # noqa: BLE001
            out["_note"] = f"scenario {i} raised {type(e).__name__}: {e}"
            return out

    out["pools_valid"] = 1.0
    out["throughput_ratio"] = float(sum(ratios) / len(ratios)) if ratios else 0.0
    return out
