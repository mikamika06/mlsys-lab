import sys
import numpy as np

sys.path.insert(0, ".")
from nmval.validator import validate_nm_constraint
from nmval.maskcheck import check_real_mask


def test_valid_2_4_sparsity():
    tensor = np.zeros((4, 4), dtype=np.float32)
    tensor[0, 0] = 1.0
    tensor[0, 1] = 1.0
    tensor[1, 2] = 1.0
    tensor[1, 3] = 1.0
    valid, counts = validate_nm_constraint(tensor, 2, 4)
    assert valid is True, f"Expected valid tensor, got counts {counts}"


def test_invalid_2_4_sparsity():
    tensor = np.ones((4, 4), dtype=np.float32)
    valid, counts = validate_nm_constraint(tensor, 2, 4)
    assert valid is False, "Expected invalid tensor due to 4 non-zeros per block"


def test_mask_checker_accuracy():
    mask = np.zeros((2, 4), dtype=bool)
    mask[0, 0] = True
    mask[0, 1] = True
    valid, counts = check_real_mask(mask, 2, 4)
    assert valid is True
    assert counts == [2, 0]
