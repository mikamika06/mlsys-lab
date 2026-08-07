import sys
sys.path.insert(0, ".")
from shimdiff.diff import find_ignored_parameter


def test_ignored_parameter_detection():
    def mock_runner(params):
        temp = params.get("temperature", 1.0)
        top_p = params.get("top_p", 1.0)
        return {"tokens": [1, 2, 3], "logprobs": [temp * 0.1, top_p * 0.2]}

    base_params = {"temperature": 0.7, "top_p": 0.9, "presence_penalty": 0.0}
    candidates = {
        "temperature": [0.1, 0.5, 0.9],
        "presence_penalty": [0.5, 1.0, 1.5],
    }

    ignored = find_ignored_parameter(mock_runner, base_params, candidates)
    assert ignored == "presence_penalty", f"expected presence_penalty, got {ignored}"


def test_no_false_positives_when_all_params_active():
    def mock_runner(params):
        return {"val": params.get("temperature", 1.0) + params.get("top_p", 1.0)}

    base_params = {"temperature": 0.7, "top_p": 0.9}
    candidates = {
        "temperature": [0.1, 0.5],
        "top_p": [0.2, 0.8],
    }

    ignored = find_ignored_parameter(mock_runner, base_params, candidates)
    assert ignored is None, f"expected None, got {ignored}"
