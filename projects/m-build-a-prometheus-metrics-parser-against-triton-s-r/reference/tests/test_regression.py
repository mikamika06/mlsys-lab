from triton_metrics.parser import parse_prometheus_text
from triton_metrics.aggregator import compute_model_request_summary, compute_gpu_utilization_summary


def test_model_summary_aggregation():
    raw_data = """
# HELP nv_inference_request_success Number of successful inference requests
# TYPE nv_inference_request_success counter
nv_inference_request_success{model="resnet50",version="1"} 100
nv_inference_request_success{model="resnet50",version="2"} 50
nv_inference_request_success{model="bert",version="1"} 200
nv_inference_compute_infer_time_us{model="resnet50",version="1"} 500000
nv_inference_compute_infer_time_us{model="resnet50",version="2"} 250000
nv_inference_compute_infer_time_us{model="bert",version="1"} 1000000
nv_inference_exec_count{model="resnet50",version="1"} 100
nv_inference_exec_count{model="resnet50",version="2"} 50
nv_inference_exec_count{model="bert",version="1"} 200
"""
    samples = parse_prometheus_text(raw_data)
    summary = compute_model_request_summary(samples)

    assert "resnet50" in summary
    assert summary["resnet50"]["success_count"] == 150.0
    assert abs(summary["resnet50"]["avg_compute_time_ms"] - 5.0) < 1e-4

    assert "bert" in summary
    assert summary["bert"]["success_count"] == 200.0
    assert abs(summary["bert"]["avg_compute_time_ms"] - 5.0) < 1e-4


def test_gpu_utilization_aggregation():
    raw_data = """
nv_gpu_utilization{gpu="0"} 0.85
nv_gpu_utilization{gpu="0"} 0.95
nv_gpu_utilization{gpu="1"} 0.40
"""
    samples = parse_prometheus_text(raw_data)
    gpu_summary = compute_gpu_utilization_summary(samples)

    assert "0" in gpu_summary
    assert abs(gpu_summary["0"] - 0.90) < 1e-4
    assert abs(gpu_summary["1"] - 0.40) < 1e-4
