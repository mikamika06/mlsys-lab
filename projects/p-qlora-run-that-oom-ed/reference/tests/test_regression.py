import sys
sys.path.insert(0, ".")
from qlora_fix.memory import account_memory
from qlora_fix.optimizer import run_training_step

def test_peak_memory_limit():
    res = account_memory(4500, 1200, 3000)
    assert res["total_mb"] <= 24000, f"Memory exceeds limit: {res['total_mb']}"

def test_effective_batch_size():
    res = run_training_step(model=None, batch=2, accum_steps=4)
    assert res["effective_batch"] == 8, f"Batch size mismatch: {res['effective_batch']}"
