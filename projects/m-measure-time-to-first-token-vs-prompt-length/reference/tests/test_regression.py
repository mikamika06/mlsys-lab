import sys
sys.path.insert(0, ".")
from ttft.metrics import extract_ttft, aggregate_runs
from ttft.model import fit_latency_model, predict_ttft
from ttft.analyze import compute_relative_error


def test_extraction_accuracy():
    logs = [{"tokens_evaluated": 128, "prompt_eval_time_ms": 45.5}]
    res = extract_ttft(logs)
    assert len(res) == 1
    assert res[0][0] == 128


def test_model_linear_fit():
    lengths = [100, 200, 300]
    times = [20.0, 40.0, 60.0]
    params = fit_latency_model(lengths, times)
    pred = predict_ttft(params, 250)
    assert abs(pred - 50.0) < 1e-5


def test_relative_error_computation():
    actual = [10.0, 20.0]
    predicted = [11.0, 19.0]
    err = compute_relative_error(actual, predicted)
    assert 0.0 <= err <= 1.0
