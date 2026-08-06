import numpy as np


def generate_records():
    rng = np.random.default_rng(42)
    frameworks = ["fw_alpha", "fw_beta", "fw_gamma"]
    configs = [f"cfg_{i}" for i in range(5)]
    gpus_list = [1, 2, 4, 8]
    records = []
    base_speed = {"fw_alpha": 1200.0, "fw_beta": 1500.0, "fw_gamma": 1000.0}
    base_vram = {"fw_alpha": 24.0, "fw_beta": 18.0, "fw_gamma": 22.0}
    scaling_mult = {"fw_alpha": 0.92, "fw_beta": 0.85, "fw_gamma": 0.95}

    for cfg in configs:
        cfg_factor = 0.8 + 0.4 * rng.uniform()
        for fw in frameworks:
            for n_gpus in gpus_list:
                eff = (scaling_mult[fw]) ** (np.log2(n_gpus))
                tps = base_speed[fw] * cfg_factor * n_gpus * eff + float(rng.normal(0, 2))
                vram = base_vram[fw] * cfg_factor * (1.0 + 0.05 * (n_gpus - 1)) + float(rng.normal(0, 0.1))
                records.append({
                    "framework": fw,
                    "config_id": cfg,
                    "num_gpus": n_gpus,
                    "tokens_per_sec": float(max(10.0, tps)),
                    "vram_gb": float(max(1.0, vram)),
                })
    return records


BENCHMARK_RECORDS = generate_records()


def compute_pairwise_ratios(records):
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


def compute_scaling_efficiency(records):
    baselines = {}
    for r in records:
        if r["num_gpus"] == 1:
            key = (r["framework"], r["config_id"])
            baselines[key] = r["tokens_per_sec"]

    effs = {}
    for r in records:
        key_base = (r["framework"], r["config_id"])
        if key_base in baselines and r["num_gpus"] > 1:
            base_tps = baselines[key_base]
            pair_key = (r["framework"], r["num_gpus"])
            eff = r["tokens_per_sec"] / (base_tps * r["num_gpus"])
            effs.setdefault(pair_key, []).append(eff)

    return {pair_key: float(np.mean(vals)) for pair_key, vals in effs.items()}
