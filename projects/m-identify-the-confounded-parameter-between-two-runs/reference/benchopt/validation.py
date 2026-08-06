def check_tg_ordering(tg_runs: list) -> bool:
    sorted_runs = sorted(tg_runs, key=lambda x: x["bytes_read"])
    tgs = [r["tg_throughput"] for r in sorted_runs]
    return tgs == sorted(tgs)
