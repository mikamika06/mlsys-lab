import json

TEST_BENCHMARKS = [
    {
        "raw_json": json.dumps([
            {"n_prompt": 512, "n_gen": 0, "avg_ts": 1500.0, "quant": "Q4_K_M"},
            {"n_prompt": 0, "n_gen": 128, "avg_ts": 60.0, "quant": "Q4_K_M"},
            {"n_prompt": 512, "n_gen": 0, "avg_ts": 1100.0, "quant": "Q8_0"},
            {"n_prompt": 0, "n_gen": 128, "avg_ts": 35.0, "quant": "Q8_0"}
        ]),
        "model_bytes": {"Q4_K_M": 4.0 * 1e9, "Q8_0": 7.0 * 1e9},
        "bandwidth_gbps": 300.0
    },
    {
        "raw_json": json.dumps([
            {"n_prompt": 1024, "n_gen": 0, "avg_ts": 2000.0, "quant": "Q4_K_M"},
            {"n_prompt": 0, "n_gen": 256, "avg_ts": 80.0, "quant": "Q4_K_M"},
            {"n_prompt": 1024, "n_gen": 0, "avg_ts": 1400.0, "quant": "Q8_0"},
            {"n_prompt": 0, "n_gen": 256, "avg_ts": 48.0, "quant": "Q8_0"}
        ]),
        "model_bytes": {"Q4_K_M": 3.8 * 1e9, "Q8_0": 6.8 * 1e9},
        "bandwidth_gbps": 400.0
    }
]


def reference_parse(raw_json_str):
    data = json.loads(raw_json_str)
    records = []
    for entry in data:
        n_prompt = entry.get("n_prompt", 0)
        n_gen = entry.get("n_gen", 0)
        avg_ts = entry.get("avg_ts", 0.0)
        quant = entry.get("ftype", entry.get("quant", "UNKNOWN"))
        mode = "pp" if n_prompt > 0 and n_gen == 0 else ("tg" if n_gen > 0 and n_prompt == 0 else "mixed")
        records.append({
            "quant": quant,
            "mode": mode,
            "n_prompt": n_prompt,
            "n_gen": n_gen,
            "tokens_per_sec": avg_ts
        })
    return records


def reference_extract(parsed_data, quant_type):
    pp, tg = 0.0, 0.0
    for row in parsed_data:
        if row["quant"] == quant_type:
            if row["mode"] == "pp":
                pp = row["tokens_per_sec"]
            elif row["mode"] == "tg":
                tg = row["tokens_per_sec"]
    return {"pp": pp, "tg": tg}


def reference_predict_tg(model_bytes, bandwidth_gbps):
    return (bandwidth_gbps * 1e9) / model_bytes


def reference_ratio(actual_tg, predicted_tg):
    return actual_tg / predicted_tg if predicted_tg > 0 else 0.0
