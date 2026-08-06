import sys

sys.path.insert(0, ".")
from layout.pipeline import steady_state_batch_time
from layout.strides import compute_nhwc_strides, is_channels_last


def test_channels_last_rejects_partial_matches():
    shape = (2, 4, 8, 8)
    good_strides = compute_nhwc_strides(shape)
    assert is_channels_last(shape, good_strides)

    bad_strides = (100, 1, 10, 2)
    assert not is_channels_last(shape, bad_strides)


def test_pipeline_respects_pin_memory():
    sync = steady_state_batch_time(10, 10, 5, pin_memory=False, non_blocking=True)
    assert sync == 20
