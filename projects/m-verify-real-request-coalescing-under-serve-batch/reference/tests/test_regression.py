import sys
sys.path.insert(0, ".")
from batching.exceptions import handle_sync_function_batch, RayServeSyncException


def test_sync_function_exception_handling():
    def bad_func(x):
        if x < 0:
            raise ValueError("negative")
        return x * 2

    res = handle_sync_function_batch(bad_func, [1, 2, 3])
    assert res == [2, 4, 6]

    try:
        handle_sync_function_batch(bad_func, [-1])
        assert False, "should have raised"
    except RayServeSyncException:
        pass
