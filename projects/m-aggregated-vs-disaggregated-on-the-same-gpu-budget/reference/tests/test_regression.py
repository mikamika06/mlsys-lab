from disagg.simulator import Request, simulate_disaggregated


def test_disaggregated_kv_transfer_impact():
    """Verify disaggregated simulation accounts for KV transfer overhead."""
    reqs = [Request(req_id=0, arrival_time=0.0, prompt_len=1000, decode_len=10)]
    
    fast_transfer = simulate_disaggregated(
        requests=reqs,
        num_prefill_gpus=1,
        num_decode_gpus=1,
        prefill_rate=1000.0,
        decode_rate=100.0,
        kv_transfer_rate=1e9,
        bytes_per_token=1024,
    )

    slow_transfer = simulate_disaggregated(
        requests=reqs,
        num_prefill_gpus=1,
        num_decode_gpus=1,
        prefill_rate=1000.0,
        decode_rate=100.0,
        kv_transfer_rate=1e5,
        bytes_per_token=1024,
    )

    assert slow_transfer[0]["ttft"] > fast_transfer[0]["ttft"], "KV transfer speed must impact TTFT"
