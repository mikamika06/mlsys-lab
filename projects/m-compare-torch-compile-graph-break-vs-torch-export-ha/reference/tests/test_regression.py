import sys
sys.path.insert(0, ".")
from graphops.branch import execute_with_cond_pattern
import ref


def test_cond_branch_shape_and_dtype():
    p, x, y = ref.get_valid_test_inputs()
    res = execute_with_cond_pattern(p, x, y)
    assert res is not None
    assert isinstance(res, float)
