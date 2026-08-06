from memattn.planner import max_sequence_length, max_batch_size


def test_lse_linear_scaling():
    b, h, d, mem = 2, 8, 64, 1073741824
    seq_lse = max_sequence_length(b, h, d, mem, mode="lse")
    seq_prob = max_sequence_length(b, h, d, mem, mode="prob")
    assert seq_lse > seq_prob * 2

    batch_lse = max_batch_size(4096, h, d, mem, mode="lse")
    batch_prob = max_batch_size(4096, h, d, mem, mode="prob")
    assert batch_lse > batch_prob
