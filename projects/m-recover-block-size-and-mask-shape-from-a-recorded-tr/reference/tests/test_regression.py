import sys
sys.path.insert(0, ".")
from triton_trace.recover import recover_block_size
from triton_trace.mask import recover_mask_shape


def test_recover_block_size_basic():
    events = [{"offset": i * 4} for i in range(32)]
    bs = recover_block_size(events)
    assert isinstance(bs, tuple)
    assert len(bs) >= 1


def test_recover_mask_shape_basic():
    mask = [[True, True, False], [True, True, False]]
    events = [{"mask": m} for m in mask]
    shape = recover_mask_shape(events)
    assert shape == (2, 2)
