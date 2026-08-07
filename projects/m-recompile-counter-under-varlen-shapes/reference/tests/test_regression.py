import sys

sys.path.insert(0, ".")
from recompile.bucketing import bucket_shape
from recompile.counter import count_recompiles
from recompile.capture import capture_decode_step


def test_bucket_shape_bounds():
    buckets = [64, 128, 256]
    assert bucket_shape(30, buckets) == 64
    assert bucket_shape(100, buckets) == 128
    assert bucket_shape(300, buckets) == 256


def test_recompile_counter_logic():
    shapes = [10, 20, 70]
    c = count_recompiles(shapes, bucket_size=64)
    assert c == 2


def test_capture_replay():
    g = capture_decode_step(lambda x: x * 2, (5,))
    assert g.replay() == 10
