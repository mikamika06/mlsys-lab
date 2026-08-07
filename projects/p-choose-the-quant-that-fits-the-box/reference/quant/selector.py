def generate_recommendation_table(configs: list, memory_limits: list) -> list:
    table = []
    for limit in memory_limits:
        best = None
        for cfg in configs:
            if cfg["peak_memory_mb"] <= limit * 1024:
                if best is None or cfg["bpw"] > best["bpw"]:
                    best = cfg
        table.append({"memory_limit_gb": limit, "recommended_recipe": best["name"] if best else "none"})
    return table

def auto_select_recipe(available_ram_gb: float, table: list) -> str:
    selected = "none"
    for row in table:
        if available_ram_gb >= row["memory_limit_gb"]:
            selected = row["recommended_recipe"]
    return selected
