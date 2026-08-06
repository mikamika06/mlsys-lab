import sys
sys.path.insert(0, ".")
from moeoffload.budget import calculate_vram, max_context_length
from moeoffload.oom import diagnose_oom, safe_allocation_limit

CONFIG = {
    "n_layers": 4,
    "hidden_dim": 512,
    "intermediate_dim": 1024,
    "n_experts": 8,
    "vocab_size": 32000,
    "n_kv_heads": 4,
    "head_dim": 64,
    "bytes_per_param": 2
}


def test_budget_invariants():
    vram_gpu = calculate_vram(CONFIG, offload_experts_to_cpu=False)
    vram_cpu = calculate_vram(CONFIG, offload_experts_to_cpu=True)
    assert vram_cpu < vram_gpu, "offloading experts should reduce VRAM"

    ctx_gpu = max_context_length(vram_gpu + 1024 * 1024 * 1024, CONFIG, False)
    ctx_cpu = max_context_length(vram_gpu + 1024 * 1024 * 1024, CONFIG, True)
    assert ctx_cpu > ctx_gpu, "CPU offload should allow larger context"


def test_oom_diagnosis_validity():
    flags = {"flash_attn": True, "cpu_offload_experts": True, "batch_size": 64}
    issues = diagnose_oom(flags, 8000, 10000)
    assert "insufficient_base_vram" in issues
    assert "flash_attn_cpu_offload_fragmentation" in issues

    limit = safe_allocation_limit(flags, 10000)
    assert limit < 10000
