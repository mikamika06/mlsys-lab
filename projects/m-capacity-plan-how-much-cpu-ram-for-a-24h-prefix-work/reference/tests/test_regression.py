from capacity.disk import measure_read_amplification

def test_read_amplification_alignment():
    block_size = 4096
    unaligned_requests = [
        {"offset_bytes": 100, "length_bytes": 500},
        {"offset_bytes": 4000, "length_bytes": 200},
    ]
    res = measure_read_amplification(block_size, unaligned_requests)
    assert res["read_amplification"] > 1.0
    assert res["physical_bytes"] == 3 * block_size
