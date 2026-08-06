from affinity.logs import parse_logs

def parse_and_compare(logs):
    avgs = parse_logs(logs)
    baselines = [avgs[l["run_id"]] for l in logs if l["affinity_mode"] == "balanced"]
    base_avg = sum(baselines) / len(baselines) if baselines else 1.0
    ratios = {}
    for run_id, avg in avgs.items():
        ratios[run_id] = avg / base_avg
    return ratios
