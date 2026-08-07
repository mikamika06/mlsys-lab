import sys
sys.path.insert(0, ".")
from runner.engine import Engine, EngineConfig, Request
from runner.queue_model import QueueModel

def test_queue_model_slot_bound():
    cfg = EngineConfig(gpu_memory_mb=3072, bytes_per_slot_mb=1024)
    model = QueueModel(cfg)
    pred = model.predict_p95_latency(num_users=8, prompt_len=32, output_len=50)
    assert pred > 3000.0, f"Expected queueing delay to push latency above 3000 ms, got {pred}"

def test_engine_queueing_delay():
    cfg = EngineConfig(gpu_memory_mb=3072, bytes_per_slot_mb=1024)
    engine = Engine(cfg)
    reqs = [Request(req_id=f"r{i}", arrival_time=0.0, prompt_len=32, output_len=50) for i in range(4)]
    metrics = engine.run_trace(reqs)
    max_q = max(m.queue_time_ms for m in metrics)
    assert max_q > 1000.0, f"Expected queueing delay for 4th request above 1000 ms, got {max_q}"
