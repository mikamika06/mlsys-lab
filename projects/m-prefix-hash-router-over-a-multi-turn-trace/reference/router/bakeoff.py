import numpy as np
from router.prefix import PrefixRouter, tokenize_into_blocks, compute_prefix_match


def simulate_trace(
    requests: list[dict],
    num_workers: int,
    max_blocks_per_worker: int,
    block_size: int,
    prefill_rate: float,
    decode_rate: float,
    policy: str = "rr",
    alpha: float = 0.5
) -> list[dict]:
    router = PrefixRouter(num_workers, max_blocks_per_worker, block_size)
    worker_available_time = [0.0] * num_workers
    results = []
    rr_idx = 0

    sorted_requests = sorted(requests, key=lambda r: r["arrival_time"])

    for req in sorted_requests:
        req_id = req["req_id"]
        tokens = req["tokens"]
        arrival = float(req["arrival_time"])
        gen_tokens = req.get("gen_tokens", 0)
        prompt_len = len(tokens)
        req_blocks = tokenize_into_blocks(tokens, block_size)

        if policy == "rr":
            chosen_worker = rr_idx % num_workers
            rr_idx += 1
            matched_blocks = compute_prefix_match(
                req_blocks, router.get_worker_blocks(chosen_worker)
            )
        elif policy == "prefix":
            chosen_worker, matched_blocks = router.route(tokens)
        elif policy == "kv_aware":
            best_w = 0
            best_score = -1e18
            for w in range(num_workers):
                m_blks = compute_prefix_match(req_blocks, router.get_worker_blocks(w))
                m_toks = min(m_blks * block_size, prompt_len)
                wait_time = max(0.0, worker_available_time[w] - arrival)
                score = alpha * m_toks - (1.0 - alpha) * (wait_time * prefill_rate)
                if score > best_score:
                    best_score = score
                    best_w = w
            chosen_worker = best_w
            matched_blocks = compute_prefix_match(
                req_blocks, router.get_worker_blocks(chosen_worker)
            )
        else:
            raise ValueError(f"Unknown policy: {policy}")

        matched_tokens = min(matched_blocks * block_size, prompt_len)
        effective_prefill = max(0, prompt_len - matched_tokens)

        start_time = max(arrival, worker_available_time[chosen_worker])
        queue_delay = start_time - arrival
        prefill_time = effective_prefill / prefill_rate if prefill_rate > 0 else 0.0
        ttft = queue_delay + prefill_time
        decode_time = gen_tokens / decode_rate if decode_rate > 0 else 0.0
        finish_time = start_time + prefill_time + decode_time

        worker_available_time[chosen_worker] = finish_time
        router.update_cache(chosen_worker, tokens)

        results.append({
            "req_id": req_id,
            "worker_id": chosen_worker,
            "arrival_time": arrival,
            "start_time": start_time,
            "finish_time": finish_time,
            "ttft": ttft,
            "matched_blocks": matched_blocks,
            "matched_tokens": matched_tokens,
            "effective_prefill": effective_prefill,
        })

    return results


def run_bakeoff(
    requests: list[dict],
    num_workers: int,
    max_blocks_per_worker: int,
    block_size: int,
    prefill_rate: float,
    decode_rate: float,
    alpha: float = 0.5
) -> dict[str, list[dict]]:
    out = {}
    for pol in ["rr", "prefix", "kv_aware"]:
        out[pol] = simulate_trace(
            requests,
            num_workers,
            max_blocks_per_worker,
            block_size,
            prefill_rate,
            decode_rate,
            policy=pol,
            alpha=alpha
        )
    return out


def compute_p95_ttft(results: list[dict]) -> float:
    if not results:
        return 0.0
    ttfts = [r["ttft"] for r in results]
    return float(np.percentile(ttfts, 95))
