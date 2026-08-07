import sys
import os

def _get_ref_module(mod_name):
    ref_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reference")
    if ref_dir not in sys.path:
        sys.path.insert(0, ref_dir)
    return __import__(f"runner.{mod_name}", fromlist=["*"])

def default_config():
    engine_mod = _get_ref_module("engine")
    return engine_mod.EngineConfig()

def get_reference_engine(config=None):
    engine_mod = _get_ref_module("engine")
    cfg = config or engine_mod.EngineConfig()
    return engine_mod.Engine(cfg)

def get_reference_bench(warmup_runs=0):
    bench_mod = _get_ref_module("bench")
    return bench_mod.LoadBench(warmup_runs=warmup_runs)

def get_reference_queue_model(config=None):
    qmodel_mod = _get_ref_module("queue_model")
    engine_mod = _get_ref_module("engine")
    cfg = config or engine_mod.EngineConfig()
    return qmodel_mod.QueueModel(cfg)

def calculate_p95(latencies):
    if not latencies:
        return 0.0
    s = sorted(latencies)
    idx = int(0.95 * len(s))
    if idx >= len(s):
        idx = len(s) - 1
    return float(s[idx])
