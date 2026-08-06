import random


def generate_fixture(seed=42):
    rng = random.Random(seed)
    dump = []
    for _ in range(12):
        bs = rng.randint(1, 8)
        q = rng.randint(50, 400)
        e = rng.randint(100, 500)
        dump.append(f'nv_inference_request_batch_size{{model="test",version="1"}} {bs}')
        dump.append(f'nv_inference_queue_duration_us{{model="test"}} {q}')
        dump.append(f'nv_inference_compute_infer_duration_us{{model="test"}} {e}')
    return "\n".join(dump)


def compute_batching_efficiency(dump_str):
    vals = []
    for line in dump_str.splitlines():
        if "nv_inference_request_batch_size" in line:
            parts = line.strip().split()
            if len(parts) == 2:
                vals.append(float(parts[1]))
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals) / 8.0)


def decompose_latency(dump_str):
    queues, execs = [], []
    for line in dump_str.splitlines():
        if "nv_inference_queue_duration_us" in line:
            parts = line.strip().split()
            if len(parts) == 2:
                queues.append(float(parts[1]))
        elif "nv_inference_compute_infer_duration_us" in line:
            parts = line.strip().split()
            if len(parts) == 2:
                execs.append(float(parts[1]))
    q_sum = sum(queues) if queues else 0.0
    e_sum = sum(execs) if execs else 0.0
    total = q_sum + e_sum
    if total == 0.0:
        return {"queue_fraction": 0.0, "exec_fraction": 0.0}
    return {"queue_fraction": q_sum / total, "exec_fraction": e_sum / total}


def diagnose_throughput_drop(baseline_dump, current_dump):
    b_dec = decompose_latency(baseline_dump)
    c_dec = decompose_latency(current_dump)
    if c_dec["queue_fraction"] > b_dec["queue_fraction"]:
        return "queueing"
    return "compute"
