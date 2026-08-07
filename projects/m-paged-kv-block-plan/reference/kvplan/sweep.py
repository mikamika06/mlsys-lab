from kvplan.planner import simulate_block_allocation
from kvplan.bench import generate_throughput_report


def run_request_rate_sweep(rate_list, block_size, page_budget):
    results = []
    for rate in rate_list:
        requests = []
        for i in range(rate * 10):
            requests.append({"action": "arrive", "seq_len": 128})

        sim = simulate_block_allocation(requests, block_size, page_budget)
        completed = sim["completed"]

        total_prompt = completed * 64
        total_gen = completed * 64
        duration = 10.0

        bench = generate_throughput_report(completed, total_prompt, total_gen, duration)

        results.append({
            "rate": rate,
            "completed": completed,
            "dropped": sim["dropped"],
            "token_throughput": bench["token_throughput"],
            "throughput_ratio": bench["throughput_ratio"],
            "utilization": sim["utilization"]
        })
    return results
