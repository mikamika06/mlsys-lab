from disagg.transfer import kv_cache_bytes, prefill_time_ms, transfer_time_ms
from disagg.sizing import decode_step_time_ms


def simulate_pipeline(requests: list[dict], num_p: int, num_d: int, model_cfg: dict, hardware_cfg: dict) -> dict:
    if not requests:
        return {
            "avg_ttft_ms": 0.0,
            "avg_tpot_ms": 0.0,
            "total_makespan_ms": 0.0,
            "p_utilization": 0.0,
            "d_utilization": 0.0,
        }
    p_free = [0.0] * num_p
    d_free = [0.0] * num_d

    total_ttft = 0.0
    total_tpot = 0.0
    p_busy = 0.0
    d_busy = 0.0
    max_finish = 0.0

    sorted_reqs = sorted(requests, key=lambda x: x["arrival_ms"])

    for req in sorted_reqs:
        arr = float(req["arrival_ms"])
        plen = int(req["prompt_len"])
        gtok = int(req["gen_tokens"])

        kb = kv_cache_bytes(
            plen,
            model_cfg["num_layers"],
            model_cfg["num_kv_heads"],
            model_cfg["head_dim"],
            model_cfg.get("dtype_bytes", 2),
        )
        p_ms = prefill_time_ms(plen, model_cfg, hardware_cfg["prefill_tflops"])
        trans_ms = transfer_time_ms(kb, hardware_cfg["bandwidth_gbps"], hardware_cfg.get("latency_ms", 0.0))
        p_total = p_ms + trans_ms

        p_idx = min(range(num_p), key=lambda i: p_free[i])
        p_start = max(arr, p_free[p_idx])
        p_end = p_start + p_total
        p_free[p_idx] = p_end
        p_busy += p_total

        ttft = p_end - arr
        total_ttft += ttft

        dec_step_ms = decode_step_time_ms(plen, model_cfg, hardware_cfg["decode_tflops"])
        d_total = dec_step_ms * gtok

        d_idx = min(range(num_d), key=lambda i: d_free[i])
        d_start = max(p_end, d_free[d_idx])
        d_end = d_start + d_total
        d_free[d_idx] = d_end
        d_busy += d_total

        tpot = dec_step_ms if gtok > 0 else 0.0
        total_tpot += tpot

        if d_end > max_finish:
            max_finish = d_end

    n = len(sorted_reqs)
    avg_ttft = total_ttft / n
    avg_tpot = total_tpot / n
    p_util = p_busy / (num_p * max_finish) if max_finish > 0 else 0.0
    d_util = d_busy / (num_d * max_finish) if max_finish > 0 else 0.0

    return {
        "avg_ttft_ms": avg_ttft,
        "avg_tpot_ms": avg_tpot,
        "total_makespan_ms": max_finish,
        "p_utilization": p_util,
        "d_utilization": d_util,
    }
