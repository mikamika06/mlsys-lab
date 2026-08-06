import torch
import flowcheck.mismatch as mismatch


def test_mismatch_exception():
    x = torch.tensor([1.0, 2.0])
    caught = False
    try:
        mismatch.trigger_mismatch(x)
    except Exception:
        caught = True
    assert caught, "expected torch.cond to raise shape/dtype mismatch error"
