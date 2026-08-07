from runner.engine import Engine, EngineConfig, RequestMetrics
from runner.bench import LoadBench

def build_slot_scaling_curve(engine_config: EngineConfig, slot_counts: list[int]) -> dict:
    bench = LoadBench(warmup_runs=0)
    curve = {}
    for slots in slot_counts:
        engine = Engine(engine_config)
        workload = bench.generate_workload(num_users=slots, prompt_len=32, output_len=100)
        res = bench.run_benchmark(engine, workload)
        curve[slots] = {
            "aggregate_tok_per_sec": res["aggregate_tok_per_sec"],
            "p95_latency_ms": res["p95_latency_ms"]
        }
    return curve

def find_knee(curve_data: dict) -> int:
    sorted_slots = sorted(curve_data.keys())
    if not sorted_slots:
        return 1
    knee = sorted_slots[0]
    for i in range(len(sorted_slots) - 1):
        s_curr = sorted_slots[i]
        s_next = sorted_slots[i + 1]
        p95_curr = curve_data[s_curr]["p95_latency_ms"]
        p95_next = curve_data[s_next]["p95_latency_ms"]
        if p95_curr > 0 and (p95_next / p95_curr) >= 1.35:
            knee = s_curr
            break
        knee = s_next
    return knee

def decompose_timing(metrics: list[RequestMetrics]) -> dict:
    if not metrics:
        return {
            "avg_queue_ms": 0.0,
            "avg_prefill_ms": 0.0,
            "avg_decode_ms": 0.0,
            "queue_pct": 0.0,
            "prefill_pct": 0.0,
            "decode_pct": 0.0
        }
    n = len(metrics)
    avg_q = sum(m.queue_time_ms for m in metrics) / n
    avg_p = sum(m.prefill_time_ms for m in metrics) / n
    avg_d = sum(m.decode_time_ms for m in metrics) / n
    total = avg_q + avg_p + avg_d
    if total > 0:
        q_pct = (avg_q / total) * 100.0
        p_pct = (avg_p / total) * 100.0
        d_pct = (avg_d / total) * 100.0
    else:
        q_pct, p_pct, d_pct = 0.0, 0.0, 0.0
    return {
        "avg_queue_ms": float(avg_q),
        "avg_prefill_ms": float(avg_p),
        "avg_decode_ms": float(avg_d),
        "queue_pct": float(q_pct),
        "prefill_pct": float(p_pct),
        "decode_pct": float(d_pct)
    }

def identify_bottleneck(engine_config: EngineConfig, active_users: int) -> str:
    max_slots = int(engine_config.gpu_memory_mb // engine_config.bytes_per_slot_mb)
    if active_users > max_slots:
        return "SLOTS"
    if active_users > engine_config.max_batch_size:
        return "BATCH"
    if active_users * engine_config.bytes_per_slot_mb > engine_config.gpu_memory_mb:
        return "MEMORY"
    return "NONE"

def optimize_config(engine_config: EngineConfig) -> EngineConfig:
    target_slots = 4
    new_bytes_per_slot = int(engine_config.gpu_memory_mb // target_slots)
    return EngineConfig(
        gpu_memory_mb=engine_config.gpu_memory_mb,
        bytes_per_slot_mb=new_bytes_per_slot,
        max_batch_size=max(engine_config.max_batch_size, target_slots),
        prefill_ms_per_tok=engine_config.prefill_ms_per_tok,
        decode_base_ms=engine_config.decode_base_ms,
        decode_per_slot_ms=engine_config.decode_per_slot_ms
    )
