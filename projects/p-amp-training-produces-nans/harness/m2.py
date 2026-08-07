import ref
from amp_fix.scaler_utils import check_autocast_promotion


def check(workdir):
    m = {"autocast_checked": 0.0}
    ops = ref.get_mock_ops()
    promoted = check_autocast_promotion(ops)
    if "matmul" in promoted and "exp" not in promoted:
        m["autocast_checked"] = 1.0
    return m
