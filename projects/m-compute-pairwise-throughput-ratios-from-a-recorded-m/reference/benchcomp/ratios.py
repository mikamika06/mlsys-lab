import numpy as np


def compute_pairwise_ratios(records):
    """Compute mean throughput ratios across matched configs for framework pairs."""
    grouped = {}
    for r in records:
        key = (r["config_id"], r["num_gpus"])
        grouped.setdefault(key, {})[r["framework"]] = r["tokens_per_sec"]

    ratios = {}
    for fw_map in grouped.values():
        fws = sorted(fw_map.keys())
        for i in range(len(fws)):
            for j in range(len(fws)):
                if i != j:
                    pair = (fws[i], fws[j])
                    val = fw_map[fws[i]] / fw_map[fws[j]]
                    ratios.setdefault(pair, []).append(val)

    return {pair: float(np.mean(vals)) for pair, vals in ratios.items()}


def rank_frameworks(records):
    """Rank frameworks by descending throughput and ascending peak VRAM."""
    by_fw = {}
    for r in records:
        fw = r["framework"]
        if fw not in by_fw:
            by_fw[fw] = {"speed": [], "vram": []}
        by_fw[fw]["speed"].append(r["tokens_per_sec"])
        by_fw[fw]["vram"].append(r["vram_gb"])

    speed_avg = {fw: np.mean(v["speed"]) for fw, v in by_fw.items()}
    vram_avg = {fw: np.mean(v["vram"]) for fw, v in by_fw.items()}

    speed_sorted = sorted(speed_avg.keys(), key=lambda k: speed_avg[k], reverse=True)
    vram_sorted = sorted(vram_avg.keys(), key=lambda k: vram_avg[k])
    return {"speed": speed_sorted, "vram": vram_sorted}
