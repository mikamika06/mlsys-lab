import json


def parse_llama_bench_json(raw_json_str):
    """Parse raw llama-bench JSON output into a normalized record dictionary."""
    data = json.loads(raw_json_str)
    records = []
    for entry in data:
        n_prompt = entry.get("n_prompt", 0)
        n_gen = entry.get("n_gen", 0)
        avg_ts = entry.get("avg_ts", 0.0)
        quant = entry.get("ftype", entry.get("quant", "UNKNOWN"))
        if n_prompt > 0 and n_gen == 0:
            mode = "pp"
        elif n_gen > 0 and n_prompt == 0:
            mode = "tg"
        else:
            mode = "mixed"
        records.append({
            "quant": quant,
            "mode": mode,
            "n_prompt": n_prompt,
            "n_gen": n_gen,
            "tokens_per_sec": avg_ts
        })
    return records


def extract_quant_metrics(parsed_data, quant_type):
    """Extract pp and tg tokens/sec for a specific quantization type."""
    pp = 0.0
    tg = 0.0
    for row in parsed_data:
        if row["quant"] == quant_type:
            if row["mode"] == "pp":
                pp = row["tokens_per_sec"]
            elif row["mode"] == "tg":
                tg = row["tokens_per_sec"]
    return {"pp": pp, "tg": tg}
