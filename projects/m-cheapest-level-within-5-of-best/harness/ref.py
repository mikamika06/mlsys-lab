import random


def generate_level_data(seed=42):
    rng = random.Random(seed)
    dataset = []
    for _ in range(20):
        n_levels = rng.randint(3, 6)
        latencies = [round(rng.uniform(10.0, 200.0), 2) for _ in range(n_levels)]
        setup_costs = [round(rng.uniform(0.1, 100.0) * (i + 1), 2) for i in range(n_levels)]
        tolerance = rng.choice([0.01, 0.05, 0.10])
        dataset.append((latencies, setup_costs, tolerance))
    return dataset


def ref_select_cheapest_level(latencies, setup_costs, tolerance=0.05):
    best_lat = min(latencies)
    threshold = best_lat * (1.0 + tolerance)
    candidates = []
    for idx, (lat, cost) in enumerate(zip(latencies, setup_costs)):
        if lat <= threshold:
            candidates.append((cost, idx))
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][1]


def generate_graph_data(seed=1337):
    rng = random.Random(seed)
    domains = ["", "ai.onnx", "com.microsoft", "org.pytorch"]
    op_types = ["Conv", "Relu", "Attention", "FusedMatMul", "Gelu"]
    graphs = []
    for _ in range(15):
        nodes = []
        for _ in range(rng.randint(10, 50)):
            dom = rng.choice(domains)
            op = rng.choice(op_types)
            if rng.random() < 0.2:
                op = f"com.microsoft:{op}"
            nodes.append({"op_type": op, "domain": dom, "name": f"node_{rng.randint(1, 1000)}"})
        graphs.append(nodes)
    return graphs


def ref_count_fused_nodes(graph_nodes):
    count = 0
    for node in graph_nodes:
        domain = node.get("domain", "")
        op_type = node.get("op_type", "")
        if domain == "com.microsoft" or op_type.startswith("com.microsoft:"):
            count += 1
    return count


def generate_cost_data(seed=2024):
    rng = random.Random(seed)
    cases = []
    for _ in range(10):
        requests = rng.randint(100, 50000)
        levels = [0, 1, 2, 3]
        online_setup = {lvl: round(rng.uniform(5.0, 500.0) * lvl, 2) for lvl in levels}
        per_req = {lvl: round(rng.uniform(5.0, 50.0) / (lvl + 1), 2) for lvl in levels}
        cases.append((requests, online_setup, per_req))
    return cases


def ref_evaluate_offline_vs_online(requests, online_setup_ms, per_request_latencies):
    res = {}
    for level, lat in per_request_latencies.items():
        setup = online_setup_ms.get(level, 0.0)
        online_total = setup + requests * lat
        offline_total = requests * lat
        res[level] = {
            "online_total": online_total,
            "offline_total": offline_total,
            "break_even_requests": (setup / lat) if lat > 0 else 0.0,
        }
    return res
