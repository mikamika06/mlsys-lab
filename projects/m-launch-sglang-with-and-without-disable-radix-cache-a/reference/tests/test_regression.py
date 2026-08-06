import sys
sys.path.insert(0, ".")
from sgl_utils.launcher import build_launch_command
from sgl_utils.metrics import compute_latency_ratio


def test_launcher_flags():
    cmd_on = build_launch_command("meta-llama/Llama-3-8B-Instruct", disable_radix_cache=False)
    assert "--disable-radix-cache" not in cmd_on
    cmd_off = build_launch_command("meta-llama/Llama-3-8B-Instruct", disable_radix_cache=True)
    assert "--disable-radix-cache" in cmd_off


def test_latency_ratio_bounds():
    ratio = compute_latency_ratio(12.5, 50.0)
    assert 0.0 < ratio < 1.0
    assert abs(ratio - 0.25) < 1e-5
