from disagg.p1d import DecodeWorker, Pipeline1P1D, PrefillWorker
from disagg.verify import verify_decode_skips_prefill


def test_decode_skips_prefill_flops():
    pw = PrefillWorker(gpu_id=0, num_layers=4, head_dim=64, num_heads=8)
    dw = DecodeWorker(gpu_id=1, num_layers=4, head_dim=64, num_heads=8)
    pipe = Pipeline1P1D(pw, dw)

    res = pipe.process_request("req_test_1", list(range(128)), decode_steps=10)
    ver = verify_decode_skips_prefill(res)

    assert ver["verified"] is True, f"Decode executed prefill compute: {ver['analysis']}"
    assert ver["analysis"]["decode_prefill_steps"] == 0


def test_kv_blocks_transferred_correctly():
    pw = PrefillWorker(gpu_id=0, num_layers=2, head_dim=32, num_heads=4)
    dw = DecodeWorker(gpu_id=1, num_layers=2, head_dim=32, num_heads=4)
    pipe = Pipeline1P1D(pw, dw)

    res = pipe.process_request("req_test_2", list(range(64)), decode_steps=5)
    assert "req_test_2" in dw.kv_store
    assert dw.kv_store["req_test_2"]["seq_len"] == 64 + 5
