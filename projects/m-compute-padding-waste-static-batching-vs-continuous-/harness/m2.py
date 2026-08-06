import ref


def check(workdir):
    from batching.simulator import simulate_batcher

    requests = ref.generate_workload(num_requests=25, seed=456)
    max_batch_size = 4
    step_time = 5.0

    ref_sim = simulate_batcher_ref(requests, max_batch_size, "static", step_time)
    got_sim = simulate_batcher(requests, max_batch_size, "static", step_time)

    ref_cont = simulate_batcher_ref(requests, max_batch_size, "continuous", step_time)
    got_cont = simulate_batcher(requests, max_batch_size, "continuous", step_time)

    out = {"simulation_metrics_matched": 0.0}

    def close(a, b, tol=1e-3):
        return abs(a - b) < tol

    static_ok = (
        close(ref_sim["total_time_ms"], got_sim.get("total_time_ms", -1)) and
        close(ref_sim["throughput_tokens_per_sec"], got_sim.get("throughput_tokens_per_sec", -1)) and
        close(ref_sim["avg_latency_ms"], got_sim.get("avg_latency_ms", -1))
    )

    cont_ok = (
        close(ref_cont["total_time_ms"], got_cont.get("total_time_ms", -1)) and
        close(ref_cont["throughput_tokens_per_sec"], got_cont.get("throughput_tokens_per_sec", -1)) and
        close(ref_cont["avg_latency_ms"], got_cont.get("avg_latency_ms", -1))
    )

    if static_ok and cont_ok:
        out["simulation_metrics_matched"] = 1.0
    else:
        out["_note"] = f"Static match: {static_ok}, Cont match: {cont_ok}"

    return out


def simulate_batcher_ref(requests, max_batch_size, mode, step_time_ms=10.0):
    reqs = sorted(requests, key=lambda x: (x["arrival_time"], x["id"]))
    request_stats = {}

    if mode == "static":
        current_time = 0.0
        idx = 0
        n = len(reqs)

        while idx < n:
            batch = reqs[idx : idx + max_batch_size]
            idx += max_batch_size

            batch_arrival = max(r["arrival_time"] for r in batch)
            start_time = max(current_time, float(batch_arrival))

            max_p = max(r["prompt_len"] for r in batch)
            max_d = max(r["decode_len"] for r in batch)

            prefill_steps = max_p
            decode_steps = max_d
            total_steps = prefill_steps + decode_steps

            first_token_time = start_time + prefill_steps * step_time_ms
            finish_time = start_time + total_steps * step_time_ms

            for r in batch:
                arr = float(r["arrival_time"])
                ttft = first_token_time - arr
                latency = finish_time - arr
                d_len = r["decode_len"]
                itl = (finish_time - first_token_time) / float(d_len) if d_len > 0 else 0.0

                request_stats[r["id"]] = {
                    "ttft_ms": float(ttft),
                    "latency_ms": float(latency),
                    "itl_ms": float(itl),
                    "finish_time_ms": float(finish_time),
                }

            current_time = finish_time

        total_time_ms = current_time

    elif mode == "continuous":
        pending = list(reqs)
        active = []
        completed = {}
        current_time = 0.0

        while pending or active:
            if not active and pending:
                current_time = max(current_time, float(pending[0]["arrival_time"]))

            while pending and pending[0]["arrival_time"] <= current_time and len(active) < max_batch_size:
                r = pending.pop(0)
                active.append({
                    "id": r["id"],
                    "arrival_time": float(r["arrival_time"]),
                    "prompt_len": r["prompt_len"],
                    "decode_len": r["decode_len"],
                    "remaining_prompt": r["prompt_len"],
                    "remaining_decode": r["decode_len"],
                    "first_token_time": None,
                    "decode_start_time": None,
                })

            if not active:
                continue

            current_time += step_time_ms

            to_remove = []
            for item in active:
                if item["remaining_prompt"] > 0:
                    item["remaining_prompt"] -= 1
                    if item["remaining_prompt"] == 0:
                        item["first_token_time"] = current_time
                        item["decode_start_time"] = current_time
                elif item["remaining_decode"] > 0:
                    item["remaining_decode"] -= 1
                    if item["remaining_decode"] == 0:
                        arr = item["arrival_time"]
                        finish = current_time
                        ttft = item["first_token_time"] - arr
                        lat = finish - arr
                        d_len = item["decode_len"]
                        itl = (finish - item["decode_start_time"]) / float(d_len) if d_len > 0 else 0.0

                        completed[item["id"]] = {
                            "ttft_ms": float(ttft),
                            "latency_ms": float(lat),
                            "itl_ms": float(itl),
                            "finish_time_ms": float(finish),
                        }
                        to_remove.append(item)

            for item in to_remove:
                active.remove(item)

        total_time_ms = current_time
        request_stats = completed

    total_tokens = sum(r["prompt_len"] + r["decode_len"] for r in reqs)
    throughput = (float(total_tokens) / (total_time_ms / 1000.0)) if total_time_ms > 0 else 0.0

    avg_lat = sum(s["latency_ms"] for s in request_stats.values()) / float(len(request_stats))
    avg_ttft = sum(s["ttft_ms"] for s in request_stats.values()) / float(len(request_stats))
    avg_itl = sum(s["itl_ms"] for s in request_stats.values()) / float(len(request_stats))

    return {
        "total_time_ms": float(total_time_ms),
        "throughput_tokens_per_sec": float(throughput),
        "avg_latency_ms": float(avg_lat),
        "avg_ttft_ms": float(avg_ttft),
        "avg_itl_ms": float(avg_itl),
        "request_stats": request_stats,
    }
