import ref
from vllmsched.latency import find_optimal_max_num_seqs as learner_find_opt
from vllmsched.analysis import quantify_hol_blocking as learner_quantify
from ref import generate_latency_test_data
import numpy as np

def ref_simulate_e2e_latencies(requests, arrival_times, max_num_seqs, max_batched_tokens, time_per_prefill_token, time_per_decode_step):
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


def ref_find_optimal_max_num_seqs(requests, arrival_times, candidate_seqs, max_batched_tokens, time_per_prefill_token, time_per_decode_step):
    best_seqs = None
    best_p99 = float("inf")

    for max_seqs in candidate_seqs:
        latencies = ref_simulate_e2e_latencies(requests, arrival_times, max_seqs, max_batched_tokens, time_per_prefill_token, time_per_decode_step)
        p99 = float(np.percentile(latencies, 99))
        if p99 < best_p99:
            best_p99 = p99
            best_seqs = max_seqs

    return best_seqs


def check(workdir):
    out = {"opt_max_seqs_matched": 0.0, "hol_delay_matched": 0.0}
    requests, arrival_times = generate_latency_test_data()
    candidate_seqs = [2, 4, 8, 16]
    max_batched_tokens = 600
    t_prefill = 0.0001
    t_decode = 0.005

    want_opt = ref_find_optimal_max_num_seqs(requests, arrival_times, candidate_seqs, max_batched_tokens, t_prefill, t_decode)
    try:
        got_opt = learner_find_opt(requests, arrival_times, candidate_seqs, max_batched_tokens, t_prefill, t_decode)
        if got_opt == want_opt:
            out["opt_max_seqs_matched"] = 1.0
        else:
            out["_note_opt"] = f"Expected opt_seqs {want_opt}, got {got_opt}"
    except Exception as e:
        out["_note_opt"] = f"Opt search failed: {type(e).__name__}: {str(e)}"

    long_prefill_len = 32768
    num_short_decodes = 100
    short_decode_len = 10
    want_hol = float(long_prefill_len * t_prefill * num_short_decodes)
    try:
        got_hol = learner_quantify(long_prefill_len, num_short_decodes, short_decode_len, t_prefill, t_decode)
        if abs(got_hol - want_hol) < 1e-5:
            out["hol_delay_matched"] = 1.0
        else:
            out["_note_hol"] = f"Expected HOL delay {want_hol}, got {got_hol}"
    except Exception as e:
        out["_note_hol"] = f"Quantify HOL failed: {type(e).__name__}: {str(e)}"

    return out
