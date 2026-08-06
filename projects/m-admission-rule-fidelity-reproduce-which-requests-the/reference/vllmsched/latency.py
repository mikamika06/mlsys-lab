import numpy as np

def simulate_e2e_latencies(requests, arrival_times, max_num_seqs, max_batched_tokens, time_per_prefill_token, time_per_decode_step):
    reqs = [dict(r) for r in requests]
    for r, arr in zip(reqs, arrival_times):
        r["arrival_time"] = arr
        r["remaining_output"] = r["output_len"]
        r["state"] = "waiting"

    current_time = 0.0
    completed = {}
    
    while len(completed) < len(reqs):
        arrived = [r for r in reqs if r["arrival_time"] <= current_time and r["id"] not in completed]
        if not arrived:
            next_arr = min(r["arrival_time"] for r in reqs if r["id"] not in completed)
            current_time = next_arr
            arrived = [r for r in reqs if r["arrival_time"] <= current_time and r["id"] not in completed]

        running = [r for r in arrived if r["state"] == "running"]
        waiting = [r for r in arrived if r["state"] == "waiting"]

        batch = []
        tokens_in_batch = 0

        for r in running:
            batch.append((r, "decode", 1))
            tokens_in_batch += 1

        for r in waiting:
            if len(batch) + 1 <= max_num_seqs and tokens_in_batch + r["prompt_len"] <= max_batched_tokens:
                batch.append((r, "prefill", r["prompt_len"]))
                tokens_in_batch += r["prompt_len"]

        if not batch:
            current_time += 0.001
            continue

        step_duration = 0.0
        for r, phase, tokens in batch:
            if phase == "prefill":
                step_duration += tokens * time_per_prefill_token
                r["state"] = "running"
            else:
                step_duration += tokens * time_per_decode_step
                r["remaining_output"] -= 1

        current_time += step_duration

        for r, phase, tokens in batch:
            if r["remaining_output"] <= 0 and r["id"] not in completed:
                completed[r["id"]] = current_time - r["arrival_time"]

    return [completed[r["id"]] for r in reqs]


def find_optimal_max_num_seqs(requests, arrival_times, candidate_seqs, max_batched_tokens, time_per_prefill_token, time_per_decode_step):
    best_seqs = None
    best_p99 = float("inf")

    for max_seqs in candidate_seqs:
        latencies = simulate_e2e_latencies(requests, arrival_times, max_seqs, max_batched_tokens, time_per_prefill_token, time_per_decode_step)
        p99 = float(np.percentile(latencies, 99))
        if p99 < best_p99:
            best_p99 = p99
            best_seqs = max_seqs

    return best_seqs
