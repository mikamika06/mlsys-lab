from batching.diagnostics import diagnose_throughput_drop
from batching.latency import decompose_latency
from batching.efficiency import compute_batching_efficiency


def test_diagnostics_queueing():
    d1 = 'nv_inference_request_batch_size{model="m"} 4\nnv_inference_queue_duration_us{model="m"} 100\nnv_inference_compute_infer_duration_us{model="m"} 100'
    d2 = 'nv_inference_request_batch_size{model="m"} 4\nnv_inference_queue_duration_us{model="m"} 900\nnv_inference_compute_infer_duration_us{model="m"} 100'
    assert diagnose_throughput_drop(d1, d2) == "queueing"


def test_diagnostics_compute():
    d1 = 'nv_inference_request_batch_size{model="m"} 4\nnv_inference_queue_duration_us{model="m"} 100\nnv_inference_compute_infer_duration_us{model="m"} 100'
    d2 = 'nv_inference_request_batch_size{model="m"} 4\nnv_inference_queue_duration_us{model="m"} 100\nnv_inference_compute_infer_duration_us{model="m"} 900'
    assert diagnose_throughput_drop(d1, d2) == "compute"
