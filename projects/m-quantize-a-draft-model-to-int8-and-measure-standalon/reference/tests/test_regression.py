import sys
sys.path.insert(0, ".")

from draft.decide import is_net_end_to_end_win


def test_int8_win_when_accuracy_preserved():
    win = is_net_end_to_end_win(
        latency_ratio=0.5,
        alpha_fp16=0.8,
        alpha_int8=0.79,
        target_latency=100.0,
        draft_fp16_latency=10.0,
        gamma=4
    )
    assert win is True


def test_int8_loss_when_accuracy_drops_sharply():
    win = is_net_end_to_end_win(
        latency_ratio=0.5,
        alpha_fp16=0.8,
        alpha_int8=0.2,
        target_latency=100.0,
        draft_fp16_latency=10.0,
        gamma=4
    )
    assert win is False


def test_int8_loss_when_no_latency_reduction():
    win = is_net_end_to_end_win(
        latency_ratio=1.0,
        alpha_fp16=0.8,
        alpha_int8=0.75,
        target_latency=100.0,
        draft_fp16_latency=10.0,
        gamma=4
    )
    assert win is False
