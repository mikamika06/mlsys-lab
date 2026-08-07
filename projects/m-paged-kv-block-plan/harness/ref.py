import math


def calculate_paged_kv_plan(seq_lens, block_size, page_budget):
    total_blocks_needed = 0
    blocks_per_seq = []
    total_tokens = sum(seq_lens)

    for length in seq_lens:
        if length == 0:
            blocks_per_seq.append(0)
            continue
        needed = math.ceil(length / block_size)
        blocks_per_seq.append(needed)
        total_blocks_needed += needed

    allocated_blocks = min(total_blocks_needed, page_budget)
    total_capacity_tokens = allocated_blocks * block_size
    waste_tokens = total_capacity_tokens - total_tokens if total_blocks_needed <= page_budget else (allocated_blocks * block_size - sum(min(l, allocated_blocks * block_size) for l in seq_lens))
    waste_tokens = max(0, waste_tokens)

    efficiency = total_tokens / total_capacity_tokens if total_capacity_tokens > 0 else 0.0

    return {
        "total_blocks_needed": total_blocks_needed,
        "allocated_blocks": allocated_blocks,
        "blocks_per_seq": blocks_per_seq,
        "waste_tokens": waste_tokens,
        "efficiency": efficiency,
        "fits_in_budget": total_blocks_needed <= page_budget
    }


def simulate_block_allocation(request_arrival_pattern, block_size, page_budget):
    active_blocks = 0
    peak_blocks = 0
    completed = 0
    dropped = 0

    for req in request_arrival_pattern:
        action = req.get("action", "arrive")
        seq_len = req.get("seq_len", 0)
        needed = math.ceil(seq_len / block_size) if seq_len > 0 else 0

        if action == "arrive":
            if active_blocks + needed <= page_budget:
                active_blocks += needed
                peak_blocks = max(peak_blocks, active_blocks)
                completed += 1
            else:
                dropped += 1
        elif action == "depart":
            active_blocks = max(0, active_blocks - needed)

    return {
        "peak_blocks": peak_blocks,
        "active_blocks": active_blocks,
        "completed": completed,
        "dropped": dropped,
        "utilization": peak_blocks / page_budget if page_budget > 0 else 0.0
    }


def generate_throughput_report(requests_completed, total_prompt_tokens, total_gen_tokens, total_time_sec):
    if total_time_sec <= 0:
        return {
            "token_throughput": 0.0,
            "prompt_throughput": 0.0,
            "generation_throughput": 0.0,
            "throughput_ratio": 0.0
        }

    total_tokens = total_prompt_tokens + total_gen_tokens
    token_throughput = total_tokens / total_time_sec
    prompt_throughput = total_prompt_tokens / total_time_sec
    gen_throughput = total_gen_tokens / total_time_sec

    baseline_throughput = 100.0
    throughput_ratio = token_throughput / baseline_throughput

    return {
        "requests_completed": requests_completed,
        "total_tokens": total_tokens,
        "token_throughput": token_throughput,
        "prompt_throughput": prompt_throughput,
        "generation_throughput": gen_throughput,
        "throughput_ratio": throughput_ratio
    }


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


TEST_CONFIGS = [
    {"seq_lens": [128, 256, 512], "block_size": 16, "page_budget": 100},
    {"seq_lens": [1, 15, 16, 17], "block_size": 16, "page_budget": 10},
    {"seq_lens": [1024, 2048], "block_size": 64, "page_budget": 30}
]
