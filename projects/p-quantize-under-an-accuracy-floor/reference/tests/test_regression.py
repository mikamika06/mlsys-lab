import sys
sys.path.insert(0, ".")
import ref
from quant.eval import run_eval
from quant.calib import get_calibration_data
from quant.recipe import quantize_uniform
from quant.mixed import assign_mixed_precision
from quant.target import check_target
from quant.report import generate_report


def test_eval_metrics():
    model = ref.ToyModel()
    x, y = ref.generate_dataset()
    res = run_eval(model, x, y)
    assert "accuracy" in res
    assert "mse" in res


def test_calibration():
    x, _ = ref.generate_dataset()
    calib = get_calibration_data(x, 16)
    assert calib.shape[0] == 16


def test_quantization():
    w = ref.ToyModel().layers[0]
    q = quantize_uniform(w, 8)
    assert q["bits"] == 8
    assert q["weights"].dtype.name == "int8"


def test_mixed_precision():
    model = ref.ToyModel()
    mp = assign_mixed_precision(model)
    assert len(mp) == len(model.layers)


def test_target_check():
    assert check_target(100, 45, 0.90, 0.895) is True
    assert check_target(100, 80, 0.90, 0.895) is False


def test_report():
    rep = generate_report({"size_ratio": 0.5, "acc_drop": 0.005})
    assert isinstance(rep, str)
