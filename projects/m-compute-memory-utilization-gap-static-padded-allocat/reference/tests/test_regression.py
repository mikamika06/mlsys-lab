from gap.utilization import compute_utilization_gap


def test_static_padding_gap_detection():
    seq_lengths = [128, 256, 512, 64]
    max_seq_len = 2048
    block_size = 16
    bytes_per_token = 1024

    res = compute_utilization_gap(seq_lengths, max_seq_len, block_size, bytes_per_token)

    assert res["static_waste_bytes"] > res["paged_waste_bytes"]
    assert res["utilization_gap_ratio"] > 0.5
