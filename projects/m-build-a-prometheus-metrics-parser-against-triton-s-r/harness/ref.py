import random
from triton_metrics.parser import MetricSample, parse_prometheus_text
from triton_metrics.aggregator import compute_model_request_summary, compute_gpu_utilization_summary


def generate_prometheus_payload(seed: int = 42) -> str:
    rng = random.Random(seed)
    models = ["resnet50", "bert_base", "llama2", "mistral"]
    gpus = ["0", "1", "2", "3"]
    lines = [
        "# HELP nv_inference_request_success Number of successful requests",
        "# TYPE nv_inference_request_success counter",
        "# HELP nv_inference_compute_infer_time_us Total compute infer time in us",
        "# TYPE nv_inference_compute_infer_time_us counter",
        "# HELP nv_gpu_utilization GPU utilization ratio",
        "# TYPE nv_gpu_utilization gauge",
    ]

    for model in models:
        for ver in ["1", "2"]:
            reqs = rng.randint(50, 1000)
            avg_lat_us = rng.uniform(1000.0, 20000.0)
            tot_lat_us = reqs * avg_lat_us
            lines.append(f'nv_inference_request_success{{model="{model}",version="{ver}"}} {reqs}')
            lines.append(f'nv_inference_compute_infer_time_us{{model="{model}",version="{ver}"}} {tot_lat_us:.1f}')
            lines.append(f'nv_inference_exec_count{{model="{model}",version="{ver}"}} {reqs}')

    for gpu in gpus:
        for _ in range(5):
            util = rng.uniform(0.1, 0.99)
            lines.append(f'nv_gpu_utilization{{gpu="{gpu}"}} {util:.4f}')

    return "\n".join(lines)
